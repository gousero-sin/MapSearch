import os
import time
import json
import pandas as pd
import requests
from geopy.distance import geodesic
from thefuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configuration
EXCEL_FILE = "Planilha_GeoLoc.xlsx"
OUTPUT_FILE = "Planilha_Validada.xlsx"
NOMINATIM_DELAY_SEC = 1.1 # Respect 1s limit
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Locks
nominatim_lock = Lock()
print_lock = Lock()

def parse_address_with_deepseek(address_raw):
    """Parses address using DeepSeek API with caching/mocking."""
    if not address_raw or pd.isna(address_raw):
        return None

    # Mock logic for testing if key is missing
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == 'your_api_key_here':
        # Simple local fallback mocks for the provided examples
        if "goianésia" in str(address_raw).lower():
            return {"street": "Avenida Brasil", "city": "Goianésia", "neighborhood": "Centro"}
        if "goiânia" in str(address_raw).lower():
            return {"street": "Avenida Ahanguera", "city": "Goiânia", "neighborhood": "Centro"}
        return {"street": "", "city": "", "neighborhood": ""}

    prompt = f'Analise o seguinte endereço e extraia logradouro (rua, avenida), bairro e cidade. Retorne APENAS um JSON válido no formato: {{"street": "...", "neighborhood": "...", "city": "..."}}. Se não encontrar algum dado, deixe vazio. Endereço: "{address_raw}"'
    
    try:
        response = requests.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            },
            timeout=15
        )
        if response.status_code != 200:
             return None

        content = response.json()['choices'][0]['message']['content']
        
        # Cleanup JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        return json.loads(content)
        
    except Exception:
        return None

def compare_addresses_with_deepseek(user_raw, nominatim_full):
    """
    Asks DeepSeek if the User Address matches the Nominatim Address semanticallly.
    Returns: (match: bool, confidence: float, reason: str)
    """
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == 'your_api_key_here':
        return False, 0.0, "No API Key"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        prompt = f"""
        Compare these two addresses. Are they the SAME location?
        
        INPUT: "{user_raw}"
        FOUND: "{nominatim_full}"
        
        TASK:
        - Return TRUE if they refer to the same place, street, or intersection.
        - Return FALSE if they are clearly different streets/cities.
        - Ignore Zip Codes.
        - Handle abbreviations (Av = Avenida).
        
        JSON OUTPUT:
        {{ "match": true/false, "confidence": 0.0-1.0, "reason": "short explanation" }}
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        return data.get('match', False), data.get('confidence', 0.0), data.get('reason', '')
    except Exception as e:
        # print(f"DeepSeek Compare Error: {e}")
        return False, 0.0, f"Error: {e}"

def get_address_from_nominatim_safe(lat, lon):
    """Thread-safe Reverse geocoding with Nominatim."""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    
    # ⚠️ CRITICAL: Ensure we only hit Nominatim once per SECOND globally
    with nominatim_lock:
        time.sleep(NOMINATIM_DELAY_SEC)
        try:
            response = requests.get(url, headers={
                'User-Agent': 'ValidatorScript_Py_Multi/3.0 (myemail@test.com)',
                'Referer': 'http://localhost'
            }, timeout=10)
            
            if response.status_code != 200:
                return None
                
            return response.json()
        except Exception:
            return None

def normalize(text):
    if not text: return ""
    return str(text).lower().strip()

def validate_row(row, index):
    """Process a single row (Thread Worker)."""
    address_full_raw = row.get('COMPLETO')
    
    try:
        # Prioritize User Input (LAT/LONG)
        lat = row.get('LAT')
        if pd.isna(lat): lat = row.get('Latitude')
        
        lon = row.get('LONG')
        if pd.isna(lon): lon = row.get('Longitude')
        
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return row, False, "Invalid Coordinates (Columns: LAT/Latitude, LONG/Longitude)", 0, 0, None, None

    if pd.isna(lat) or pd.isna(lon):
        return row, False, "Missing Coordinates", 0, 0, None, None
        
    if pd.isna(address_full_raw):
        # Try alternatives
        address_full_raw = row.get('Maps_Address') or row.get('Address') or row.get('Endereço')

    if pd.isna(address_full_raw):
        return row, False, "Missing Address Data", 0, 0, None, None

    # --- STEP 1: DeepSeek Parsing (Parallel) ---
    parsed_address = parse_address_with_deepseek(address_full_raw)
    parsed_street = normalize(parsed_address.get('street', '')) if parsed_address else ""
    parsed_city = normalize(parsed_address.get('city', '')) if parsed_address else ""

    # --- STEP 2: Nominatim Geocoding (Serialized) ---
    nom_result = get_address_from_nominatim_safe(lat, lon)
    
    if not nom_result or 'address' not in nom_result:
        return row, False, "Geocoding Failed", 0, 0, None, parsed_address

    nom_address_obj = nom_result.get('address', {})
    nom_display_name = nom_result.get('display_name', '')
    nom_lat = float(nom_result['lat'])
    nom_lon = float(nom_result['lon'])

    # --- STEP 3: Validation Logic ---
    dist_m = geodesic((lat, lon), (nom_lat, nom_lon)).meters
    
    # Fuzzy Match: Input vs Nominatim
    input_norm = normalize(address_full_raw)
    nom_norm = normalize(nom_display_name)
    match_score = fuzz.token_set_ratio(input_norm, nom_norm)

    # DeepSeek Comparison: Parsed Street vs Nominatim Road
    nom_road = normalize(nom_address_obj.get('road', ''))
    deepseek_street_match = fuzz.token_set_ratio(parsed_street, nom_road) if parsed_street else 0

    # City Check
    nom_city_val = normalize(nom_address_obj.get('city') or nom_address_obj.get('town') or nom_address_obj.get('municipality') or "")
    city_matches = False
    if nom_city_val and (nom_city_val in input_norm or (parsed_city and parsed_city in nom_city_val)):
        city_matches = True
    elif nom_city_val and fuzz.partial_ratio(nom_city_val, input_norm) > 85:
        city_matches = True

    is_valid = False
    reasons = []

    # --- DeepSeek Semantic Check ---
    ds_match = False
    ds_conf = 0.0
    ds_reason = ""
    
    # Only call DeepSeek if we have data. 
    # Use ThreadPool logic or sequential? It's inside a function running in ThreadPoolExecutor already.
    # Calling API synchronously here is fine (thread blocked).
    if address_full_raw and nom_display_name:
         ds_match, ds_conf, ds_reason = compare_addresses_with_deepseek(address_full_raw, nom_display_name)

    # Logic Implementation
    if ds_match and ds_conf > 0.6:
         is_valid = True
         reasons.append(f"DeepSeek Semantic Match ({int(ds_conf*100)}%): {ds_reason}")
    elif match_score >= 80:
         is_valid = True
         reasons.append(f"High Fuzzy Match ({match_score})")
    elif dist_m < 50 and city_matches:
         is_valid = True
         reasons.append("High Precision (<50m)")
    elif dist_m < 600 and city_matches and (match_score > 40 or deepseek_street_match > 70):
         # DeepSeek helps here: if input text is messy, but DeepSeek extracted street matches Nominatim road -> Valid!
         is_valid = True
         reasons.append(f"Prox < 600m + Match (DS:{deepseek_street_match} / Raw:{match_score})")

    # Logging
    with print_lock:
        status_icon = "✅" if is_valid else "❌"
        # print(f"{status_icon} Row {index}: Dist: {dist_m:.1f}m | Score: {match_score} | DS-Score: {deepseek_street_match} | City: {city_matches}")

    return row, is_valid, ", ".join(reasons), dist_m, match_score, nom_display_name, parsed_address

def main():
    print("Starting Validator Script (Multithreaded 10x + DeepSeek)...")
    
    # Dependencies check
    try:
        import thefuzz
    except ImportError:
        print("Please install 'thefuzz': pip install thefuzz")
        return

    if not os.path.exists(EXCEL_FILE):
        print(f"Creating dummy file {EXCEL_FILE}...")
        df_dummy = pd.DataFrame([
            {"PARCEIRO": "Test1", "COMPLETO": "Rua da Consolação, São Paulo", "LAT": -23.5489, "LONG": -46.6388},
            {"PARCEIRO": "Test2", "COMPLETO": "Av Paulista, São Paulo", "LAT": -10.0, "LONG": -10.0}
        ])
        df_dummy.to_excel(EXCEL_FILE, index=False)

    try:
        df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Could not read excel: {e}")
        return

    results = []
    
    print(f"Processing {len(df)} rows with 10 threads...")
    
    from tqdm import tqdm
    
    # Parallel Processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(validate_row, row, i+1): i for i, row in df.iterrows()}
        
        # tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(df), desc="Validating", unit="row"):
            try:
                # Unpack result
                orig_row, is_valid, reason, dist, score, nom_addr, parsed = future.result()
                
                # Visual Feedback
                if is_valid:
                    tqdm.write(f"✅ FOUND: {nom_addr[:40]}... (Dist: {dist:.1f}m)")
                else:
                    tqdm.write(f"❌ ERROR: {reason} (Dist: {dist:.1f}m)")

                row_dict = orig_row.to_dict()
                row_dict['VALIDATION_STATUS'] = "VALID" if is_valid else "INVALID"
                row_dict['VALIDATION_REASON'] = reason
                row_dict['DISTANCE_METERS'] = round(dist, 2)
                row_dict['MATCH_SCORE'] = score
                # DeepSeek Columns
                row_dict['DS_STREET'] = parsed.get('street') if parsed else ""
                row_dict['DS_CITY'] = parsed.get('city') if parsed else ""
                row_dict['NOMINATIM_ADDRESS'] = nom_addr
                
                results.append(row_dict)
            except Exception as exc:
                tqdm.write(f"⚠️ Exception: {exc}")

    # Sort results by original index if needed, but append order is fine usually.
    # To keep original order, we might need to store index in results and sort.
    
    if results:
        # Save Excel
        df_out = pd.DataFrame(results)
        df_out.to_excel(OUTPUT_FILE, index=False)
        
        # Save JSON for Frontend
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Done! All data written to {OUTPUT_FILE} and results.json")
        valid_count = df_out[df_out['VALIDATION_STATUS'] == 'VALID'].shape[0]
        print(f"Summary: {valid_count} Valid, {len(df_out) - valid_count} Invalid.")
    else:
        print("No rows processed.")

if __name__ == "__main__":
    main()
