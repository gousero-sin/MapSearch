
import asyncio
import pandas as pd
import re
from playwright.async_api import async_playwright
import argparse
import sys
import urllib.parse
import difflib
import unicodedata
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.asyncio import tqdm

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = None
if DEEPSEEK_API_KEY:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def normalize_text(text):
    """Normalize text by removing accents and lowercasing."""
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower()


# Configure headers and columns based on user request (no header in input)
# Input columns: A=ID, B=Nome Fantasia, C=Razao Social, D=Completo
# Output columns: A=ID, B=Nome Fantasia, C=Razao Social, D=Completo, E=Latitude, F=Longitude, G=Maps Title, H=Maps Address, I=Maps URL

INPUT_FILE = 'Planilha_GeoLoc.xlsx'
OUTPUT_FILE = 'Planilha_GeoLoc_Updated.xlsx'

def verify_match_with_deepseek(input_data, found_data):
    """
    Asks DeepSeek to compare the Input data with the Found data.
    Returns: (is_match (bool), reason (str), score (float))
    """
    if not client: return False, "No API Key", 0.5

    prompt = f"""
    You are a validation engine. Compare the User Input against the Found Google Maps Result.
    
    USER INPUT:
    Name: {input_data.get('name')}
    Address: {input_data.get('address')}
    City: {input_data.get('city')}
    Full String: {input_data.get('full')}

    FOUND RESULT:
    Title: {found_data.get('title')}
    Address: {found_data.get('address')}
    Category: {found_data.get('category')}
    
    TASK:
    Determine if this is the SAME place (True) or a Different place (False).
    
    GUIDELINES:
    1. **Location Match**: If Street + Number + City match, it is a strong candidate.
    2. **Name Evaluation**:
       - **MATCH**: Aliases (e.g. "Store X" == "Rede Store"), containment ("Store X" inside "Shopping Y"), or typo fixes.
       - **MISMATCH**: Clearly different businesses at same address (e.g. "Burger King" vs "McDonalds"). 
       - **MISMATCH**: Generic City/State result vs Specific Address input.
    3. **Shopping Centers**: Input "Store - Mall X" matching "Mall X" is acceptable (user wants the location of the store, which is the mall).
    
    OUTPUT JSON:
    {{
        "match": true/false,
        "confidence": 0.0 to 1.0,
        "reason": "concise explanation"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        return data.get('match', False), data.get('reason', ''), data.get('confidence', 0.0)
    except Exception as e:
        print(f"  [DeepSeek] Validation Error: {e}")
        return False, f"Error: {e}", 0.0

async def extract_google_maps_data(page):
    """Extracts visible data from the current Google Maps place page."""
    data = {'title': '', 'address': '', 'category': ''}
    try:
        # Title
        h1 = await page.query_selector('h1')
        if h1: data['title'] = await h1.inner_text()
        
        # Address
        # Try finding the address button
        addr_btn = await page.query_selector('button[data-item-id*="address"]')
        if addr_btn:
             data['address'] = await addr_btn.inner_text()
        else:
             # Fallback
             els = await page.query_selector_all('div.Io6YTe')
             for el in els:
                 txt = await el.inner_text()
                 if ',' in txt and any(c.isdigit() for c in txt):
                     data['address'] = txt
                     break
        
        # Category (often secondary text near title)
        cat_btn = await page.query_selector('button[jsaction*="category"]')
        if cat_btn: data['category'] = await cat_btn.inner_text()
        
    except: pass
    return data

def parse_with_deepseek(text):
    """
    Uses DeepSeek API to parse unstructured address text.
    Returns dict: {name, address, number, city, state, full}
    """
    if not client: return None
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an address parser. Extract the following fields from the input text: 'business_name', 'street', 'number', 'city', 'state'. Return ONLY a valid JSON object. If a field is missing, use empty string."},
                {"role": "user", "content": f"Parse this: {text}"}
            ],
            stream=False
        )
        content = response.choices[0].message.content
        # Clean potential markdown code blocks
        content = content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        
        # Normalize keys to match our internal schema
        return {
            'full': text,
            'name': data.get('business_name', ''),
            'address': f"{data.get('street', '')}, {data.get('number', '')}".strip(', '),
            'street_number': data.get('number', ''),
            'city': data.get('city', ''),
            'state': data.get('state', '')
        }
    except Exception as e:
        print(f"  [DeepSeek] Error parsing: {e}")
        return None

def parse_input_data(input_full):
    """
    Attempts to separate Business Name from Address and extracts City/State.
    Returns a dict with: 'name', 'address', 'city', 'state', 'full'
    """
    # Try DeepSeek first
    ds_result = parse_with_deepseek(input_full)
    if ds_result:
        print("  [DeepSeek] Parsing successful.")
        return ds_result
    
    # Fallback to Regex / Hueristic
    input_norm = str(input_full).strip()
    result = {
        'full': input_norm,
        'name': '',
        'address': '',
        'city': '',
        'state': '',
        'street_number': ''
    }

    # 1. Try to split by comma
    parts = [p.strip() for p in input_norm.split(',')]
    
    # Heuristic: If we have > 2 parts, usually: Name, Street, Number, ... City, State
    # OR: Name, Address (with numbers inside), City
    
    if len(parts) >= 2:
        # Assume first part is Name, rest is Address
        result['name'] = parts[0]
        result['address'] = ", ".join(parts[1:])
        
        # Try to extract city/state from the end
        # Last part might be State (2 chars) or Country or "City - State"
        last_part = parts[-1]
        
        # Check if last matches State (2 chars)
        state_match = re.search(r'\b([A-Z]{2})\b', last_part)
        if state_match:
            result['state'] = state_match.group(1)
        
        # City is likely the second-to-last or third-to-last
        if len(parts) >= 3:
            # Often Format: ..., City, State
            if len(parts[-2]) > 3: # Avoid grabbing "N 100" as city
                result['city'] = parts[-2]
            elif len(parts) >= 4:
                result['city'] = parts[-3]
    else:
        # Fallback for no commas
        result['name'] = input_norm
    
    # 2. Extract Street Number from Address (for validation)
    # Search for isolated numbers
    # We look in the full string minus the name to avoid numbers in business name (e.g. "Club 88")
    address_part = result['address'] if result['address'] else input_norm
    number_match = re.search(r'\b(\d+)\b', address_part)
    if number_match:
        result['street_number'] = number_match.group(1)

    return result

def validate_result(parsed_input, found_title, found_address):
    """
    Validates the found result against the parsed input.
    Returns a score between 0.0 and 1.0.
    """
    if not found_title and not found_address:
        return 0.0
    
    score = 0.0
    
    # Normalize
    input_name = normalize_text(parsed_input['name'])
    input_city = normalize_text(parsed_input['city'])
    input_full = normalize_text(parsed_input['full'])
    
    found_title_norm = normalize_text(found_title)
    found_addr_norm = normalize_text(found_address)
    found_full = f"{found_title_norm} {found_addr_norm}"

    # --- CHECK 1: GENERIC CITY MATCH (CRITICAL) ---
    # If the found address is JUST "City, State", it's likely a generic center point.
    # e.g. "São Paulo, SP" or "Campinas - SP, 13000-000"
    # Rule: If found address has NO street number and very short length relative to city size...
    
    is_generic = False
    if found_addr_norm.count(',') <= 1 and not any(char.isdigit() for char in found_addr_norm):
         # Valid: "Rua X, Campinas" (has comma, maybe no number)
         # Invalid: "Campinas, SP" (short, no number)
         is_generic = True
    
    if is_generic:
        print(f"    [Validation] Warning: Result looks like generic city center: '{found_address}'")
        # Start with penalty, unless Title matches PERFECTLY
        score = 0.2
    
    # --- CHECK 2: NAME MATCH ---
    # Similarity ratio
    name_ratio = difflib.SequenceMatcher(None, input_name, found_title_norm).ratio()
    if name_ratio > 0.8:
        score = max(score, 0.9)
    elif name_ratio > 0.5:
        score = max(score, 0.6)
    
    # --- CHECK 3: STREET NUMBER MATCH ---
    # If we expected a number but found none (or wrong one), penalty
    expected_number = parsed_input['street_number']
    if expected_number:
        if expected_number in found_addr_norm or expected_number in found_title_norm:
            score += 0.2 # Boost
            # If we had a generic penalty but found the number, we recover
        else:
            # If we expected a number and the result has a DIFFERENT number, huge penalty
            # Check for ANY number in result
            found_numbers = re.findall(r'\b(\d+)\b', found_addr_norm)
            if found_numbers:
                pass
    
    # --- CHECK 4: CITY MATCH ---
    if input_city and len(input_city) > 3:
        if input_city not in found_addr_norm and input_city not in found_title_norm:
             print(f"    [Validation] City mismatch! Input: {input_city} not found in '{found_addr_norm}'")
             score *= 0.3 # Heavy penalty for wrong city
        else:
             # Confirms we are at least in the right city
             if score < 0.5: score = 0.5

    # Cap score
    return min(1.0, score)

async def search_google_maps(page, query):
    try:
        print(f"Searching for: {query}")
        await page.goto(f'https://www.google.com/maps/search/{query}', timeout=30000)
        
        # 1. DIRECT HIT CHECK
        # Wait a moment for potential redirect to /place/
        await page.wait_for_timeout(3000)
        
        is_direct_hit = False
        if '/maps/place/' in page.url:
             is_direct_hit = True
             print("  Direct Hit detected via URL.")
        else:
            try:
                 # Check for H1 presence if URL didn't change (sidebar result)
                 if await page.query_selector('h1'):
                     is_direct_hit = True
                     print("  Direct Hit detected via H1.")
            except: pass

        # 2. LIST VIEW CHECK (Only if not a direct hit)
        if not is_direct_hit:
            print("  URL/Page suggests search list. Checking for results to click...")
            try:
                # 1. Look for the main result link. Google Maps often uses a specific structure.
                # Try multiple selectors for the first result
                potential_selectors = [
                    'a[href*="/maps/place/"]',
                    'div[role="feed"] > div > div > a',
                    'a.hfpxzc' # Common class for overlay link
                ]
                
                found_result = None
                for selector in potential_selectors:
                    try:
                        # Don't wait too long per selector
                        results = await page.query_selector_all(selector)
                        if results:
                            # Filter results that are visible
                            for res in results:
                                if await res.is_visible():
                                    found_result = res
                                    break
                        if found_result: break
                    except: continue

                if found_result:
                    print(f"  Found results using selector '{selector}'. Clicking first visible...")
                    # Dispatch click is safest
                    await found_result.dispatch_event('click')
                    try:
                        await page.wait_for_load_state('networkidle', timeout=5000)
                    except:
                        await page.wait_for_timeout(3000)
                else:
                    print("  No place results found to click.")
                    return None, None, {}, page.url
            except Exception as e:
                print(f"  Error clicking result: {e}")
                return None, None, {}, page.url

        # 3. DATA EXTRACTION
        # Wait for data to load - Critical step
        try:
             await page.wait_for_selector('h1', timeout=5000)
        except: pass

        current_url = page.url
        page_data = await extract_google_maps_data(page)
        
        # Coordinates Extraction Logic
        # Priority 1: !3d / !4d (Pin Location - Most Accurate)
        # Priority 2: @lat,lon (Viewport - Only if !3d missing AND we are on a place page)
        
        lat, lon = None, None
        match_3d = re.search(r'!3d([-0-9.]+)', current_url)
        match_4d = re.search(r'!4d([-0-9.]+)', current_url)
        
        if match_3d and match_4d:
            lat = match_3d.group(1)
            lon = match_4d.group(1)
            print(f"    [Coords] Found Pin Location (!3d): {lat}, {lon}")
        else:
            # Only fall back to viewport if we strictly DO NOT have 3d/4d
            # And arguably, if we don't have a pin, maybe we shouldn't guess?
            # User said: "só deve ser pego antes se for direto pro endereço"
            # If we clicked a result, we expect specific coords.
            print("    [Coords] No Pin Location (!3d) found in URL.")
            # Check if it looks like a place page (or we detected it earlier)
            is_place_page = '/maps/place/' in current_url or (is_direct_hit)
            
            if is_place_page:
                 coord_match = re.search(r'@([-0-9.]+),([-0-9.]+)', current_url)
                 if coord_match:
                     lat = coord_match.group(1)
                     lon = coord_match.group(2)
                     print(f"    [Coords] Using Viewport Fallback (@): {lat}, {lon}")

        return lat, lon, page_data, current_url

    except Exception as e:
        print(f"Error search_google_maps: {e}")
        return None, None, {}, page.url

from tqdm import tqdm

async def process_row(index, row, context, semaphore):
    async with semaphore:
        page = await context.new_page()
        try:
            raw_input = row['Completo_Internal']
            if pd.isna(raw_input):
                return index, {k: None for k in ['Latitude', 'Longitude', 'Maps_Title', 'Maps_Address', 'Maps_URL', 'Validation_Score', 'Search_Method']}

            # Initialize row_res
            row_res = {
                'Latitude': None, 'Longitude': None, 
                'Maps_Title': None, 'Maps_Address': None, 'Maps_URL': None, 
                'Validation_Score': 0.0, 'Search_Method': None
            }

            print(f"\n--- Row {index+1} ---")
            print(f"Input: {raw_input}")
            
            parsed = parse_input_data(raw_input)
            
            # Strategy Priority:
            # 1. Full Input (Best if it works)
            # 2. Address Only (If name is confusing/outdated)
            # 3. Name + City (If address is wrong/moved, but business is known)
            
            best_result = None
            best_score = -1.0
            
            # ATTEMPT 1: Search Full Query
            lat, lon, page_data, url = await search_google_maps(page, parsed['full'])
            
            # Validation
            score = 0.0
            title = page_data.get('title', '')
            addr = page_data.get('address', '')
            
            if client and title:
                is_match, reason, conf = verify_match_with_deepseek(parsed, page_data)
                print(f"    [DeepSeek Val] Match: {is_match} | Conf: {conf} | Reason: {reason}")
                if is_match:
                    score = conf
                else:
                    score = 0.1 # Very low score if LLM says no match
            else:
                 # Fallback to local validation
                 score = validate_result(parsed, title, addr)
            
            method = "Full_Query"
            
            print(f"  Result 1 (Full): Score={score:.2f} | Title={title} | Address={addr}")
            
            best_result = (lat, lon, title, addr, url, score, "Full_Query")
            best_score = score

            # ATTEMPT 2: If score is low (e.g., < 0.65), try Address Only
            # This handles cases where business name in list is different or outdated
            if best_score < 0.65:
                if parsed['address']:
                    # Try Address + City (to ensure we don't search 'Rua 1' globally)
                    query = f"{parsed['address']}, {parsed['city']}, {parsed['state']}"
                    
                    print(f"  [Retry] Low score. Trying Address Only: {query}")
                    lat2, lon2, page_data2, url2 = await search_google_maps(page, query)
                    
                    title2 = page_data2.get('title', '')
                    addr2 = page_data2.get('address', '')
                    
                    if client and title2:
                        is_match2, reason2, conf2 = verify_match_with_deepseek(parsed, page_data2)
                        print(f"    [DeepSeek Val] Match: {is_match2} | Conf: {conf2} | Reason: {reason2}")
                        score2 = conf2 if is_match2 else 0.1
                    else:
                        score2 = validate_result(parsed, title2, addr2)
                    
                    print(f"  Result 2 (Addr): Score={score2:.2f} | Title={title2} | Address={addr2}")
                    
                    if score2 > best_score:
                        print("  -> Address search was better. Keeping it.")
                        best_result = (lat2, lon2, title2, addr2, url2, score2, "Address_Only")
                        best_score = score2
                    else:
                        print("  -> Address search was worse or same. Keeping Full search.")
            
            # ATTEMPT 3: Name + City
            if best_score < 0.65:
                 if parsed['name'] and parsed['city']:
                     query = f"{parsed['name']} {parsed['city']} {parsed['state']}"
                     print(f"  [Retry] Low score. Trying Name + City: {query}")
                     
                     lat3, lon3, page_data3, url3 = await search_google_maps(page, query)
                     
                     title3 = page_data3.get('title', '')
                     addr3 = page_data3.get('address', '')
                     
                     if client and title3:
                        is_match3, reason3, conf3 = verify_match_with_deepseek(parsed, page_data3)
                        print(f"    [DeepSeek Val] Match: {is_match3} | Conf: {conf3} | Reason: {reason3}")
                        score3 = conf3 if is_match3 else 0.1
                     else:
                        score3 = validate_result(parsed, title3, addr3)

                     print(f"  Result 3 (Name+City): Score={score3:.2f} | Title={title3} | Address={addr3}")

                     if score3 > best_score:
                        best_score = score3
                        best_result = (lat3, lon3, title3, addr3, url3, score3, "Name_City")

            final_lat, final_lon, final_title, final_addr, final_url, final_score, final_method = best_result
            
            # QUALITY CONTROL
            if final_score < 0.6:
                 print(f"  [Row {index+1}] -> Low confidence ({final_score:.2f}). Discarding coordinates.")
                 final_lat = None
                 final_lon = None
            
            row_res['Latitude'] = final_lat
            row_res['Longitude'] = final_lon
            row_res['Maps_Title'] = final_title
            row_res['Maps_Address'] = final_addr
            row_res['Maps_URL'] = final_url
            row_res['Validation_Score'] = final_score
            row_res['Search_Method'] = final_method
            
            await page.close()
            return index, row_res
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            await page.close()
            return index, {
                'Latitude': None, 'Longitude': None, 
                'Maps_Title': "ERROR", 'Maps_Address': str(e), 'Maps_URL': None, 
                'Validation_Score': 0.0, 'Search_Method': "Error"
            }

async def main(limit=None):
    try:
        df = pd.read_excel(INPUT_FILE, header=0)
        
        # Identify address column
        target_col = None
        for col in df.columns:
            if str(col).lower() == 'completo':
                target_col = col
                break
        if not target_col:
            if len(df.columns) >= 4:
                target_col = df.columns[3]
            else:
                print("Error: Could not find 'Completo' column.")
                return 

        print(f"Using column '{target_col}' for input.")
        df['Completo_Internal'] = df[target_col]
        
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    rows_to_process = df.shape[0]
    if limit:
        rows_to_process = min(int(limit), rows_to_process)
        print(f"Limit: {rows_to_process}")
        
    # Limit Concurrency to 3
    MAX_CONCURRENT_PAGES = 3
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale='pt-BR')
        
        tasks = []
        for index, row in df.iterrows():
            if index >= rows_to_process: break
            tasks.append(process_row(index, row, context, semaphore))
            
        print(f"Starting execution with {MAX_CONCURRENT_PAGES} threads...")
        
        results_list = []
        pbar = tqdm(total=len(tasks), desc="Processing", unit="row")
        
        for future in asyncio.as_completed(tasks):
            res = await future
            results_list.append(res)
            pbar.update(1)
            
        pbar.close()
        
        await browser.close()
        
        # Aggregate results
        res_map = {idx: res for idx, res in results_list}
        
        # Initialize output columns
        for col in ['Latitude', 'Longitude', 'Maps_Title', 'Maps_Address', 'Maps_URL', 'Validation_Score', 'Search_Method']:
            df[col] = None
            
        for idx, res in res_map.items():
             for k, v in res.items():
                 df.at[idx, k] = v

    # Save
    if 'Completo_Internal' in df.columns:
        df = df.drop(columns=['Completo_Internal'])

    df.iloc[:rows_to_process].to_excel(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', help='Limit rows')
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
