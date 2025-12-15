"""
FastAPI Backend for GeoValidator Dashboard
Provides endpoints to run validator.py and search_maps.py with progress streaming.
"""

import os
import json
import asyncio
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ============ App Setup ============
app = FastAPI(title="GeoValidator API", version="1.0.0")

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Storage ============
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking
jobs_file = Path("jobs.json")

def load_jobs():
    if jobs_file.exists():
        try:
            with open(jobs_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_jobs():
    with open(jobs_file, "w") as f:
        json.dump(jobs, f, indent=2)

jobs = load_jobs()  # Load on startup

# WebSocket connections for progress updates
ws_connections: dict[str, WebSocket] = {}

# ============ Helpers ============
def generate_output_filename(original_name: str, suffix: str) -> str:
    """Generate standardized output filename."""
    base = Path(original_name).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"{base}_{suffix}_{timestamp}.xlsx"

async def send_progress(job_id: str, progress: int, message: str = ""):
    """Send progress update to connected WebSocket client."""
    if job_id in ws_connections:
        try:
            await ws_connections[job_id].send_json({
                "type": "progress",
                "progress": progress,
                "message": message
            })
        except:
            pass

async def run_validator_script(input_path: str, output_path: str, job_id: str):
    """Run the validator script with progress updates."""
    import pandas as pd
    from validator import validate_row
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        df = pd.read_excel(input_path)
        total = len(df)
        results = []
        completed = 0
        
        await send_progress(job_id, 0, f"Starting validation of {total} rows...")
        
        # Nominatim limitation is 1 req/sec. DeepSeek is faster.
        # We increase threads to fill the gaps while threads wait for Nominatim lock.
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(validate_row, row, i+1): i for i, row in df.iterrows()}
            
            for future in as_completed(futures):
                try:
                    orig_row, is_valid, reason, dist, score, nom_addr, parsed = future.result()
                    
                    row_dict = orig_row.to_dict()
                    row_dict['VALIDATION_STATUS'] = "VALID" if is_valid else "INVALID"
                    row_dict['VALIDATION_REASON'] = reason
                    row_dict['DISTANCE_METERS'] = round(dist, 2)
                    row_dict['MATCH_SCORE'] = score
                    row_dict['DS_STREET'] = parsed.get('street') if parsed else ""
                    row_dict['DS_CITY'] = parsed.get('city') if parsed else ""
                    row_dict['NOMINATIM_ADDRESS'] = nom_addr
                    results.append(row_dict)
                except Exception as e:
                    pass
                
                if jobs[job_id].get("cancelled"):
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise Exception("Cancelled by user")

                completed += 1
                progress = int((completed / total) * 100)
                await send_progress(job_id, progress, f"Validated {completed}/{total}")
        
        # Save results (partial if cancelled or full)
        df_out = pd.DataFrame(results)
        df_out.to_excel(output_path, index=False)
        
        # Also save JSON - Clean text to avoid NaN (which breaks browser JSON.parse)
        json_path = output_path.replace('.xlsx', '.json')
        # Use pandas to_json to reliably handle NaNs (converts to null)
        results_json = json.loads(df_out.to_json(orient='records', date_format='iso'))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
        
        await send_progress(job_id, 100, "Validation complete!")
        return json_path
        
    except Exception as e:
        # If explicitly cancelled exception
        if "Cancelled" in str(e):
             await send_progress(job_id, -1, "Process cancelled by user.")
             # We should still save what we have? Maybe.
             return None
        await send_progress(job_id, -1, f"Error: {str(e)}")
        raise e

async def run_search_script(input_path: str, output_path: str, job_id: str, chain_validation: bool = False):
    """Run the search_maps script with progress updates, optionally chaining validation."""
    import pandas as pd
    from playwright.async_api import async_playwright
    from search_maps import process_row, parse_input_data
    
    try:
        df = pd.read_excel(input_path, header=0)
        
        # Find the address column
        target_col = None
        for col in df.columns:
            if str(col).lower() == 'completo':
                target_col = col
                break
        if not target_col and len(df.columns) >= 4:
            target_col = df.columns[3]
        
        if not target_col:
            raise ValueError("Could not find 'Completo' column")
        
        df['Completo_Internal'] = df[target_col]
        total = len(df)
        
        await send_progress(job_id, 0, f"Starting Google Maps search for {total} rows...")
        
        # Limit concurrency
        MAX_CONCURRENT = 3
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(locale='pt-BR')
            
            async def process_with_progress(index, row):
                result = await process_row(index, row, context, semaphore)
                return result
            
            tasks = [asyncio.create_task(process_with_progress(i, row)) for i, row in df.iterrows()]
            results_list = []
            completed = 0
            
            try:
                for coro in asyncio.as_completed(tasks):
                    if jobs[job_id].get("cancelled"):
                        for t in tasks:
                            t.cancel()
                        raise Exception("Cancelled by user")

                    result = await coro
                    results_list.append(result)
                    completed += 1
                    progress = int((completed / total) * 100)
                    await send_progress(job_id, progress, f"Searched {completed}/{total}")
            except asyncio.CancelledError:
                raise Exception("Cancelled by user")
            except Exception as e:
                # Cancel pending tasks if other error
                for t in tasks:
                    if not t.done(): t.cancel()
                raise e
            
            await browser.close()
        
        # Aggregate results
        res_map = {idx: res for idx, res in results_list}
        
        for col in ['Latitude', 'Longitude', 'Maps_Title', 'Maps_Address', 'Maps_URL', 'Validation_Score', 'Search_Method']:
            df[col] = None
        
        for idx, res in res_map.items():
            for k, v in res.items():
                df.at[idx, k] = v
        
        # Clean up
        if 'Completo_Internal' in df.columns:
            df = df.drop(columns=['Completo_Internal'])
        
        df.to_excel(output_path, index=False)
        
        # Also save JSON - Clean text to avoid NaN
        json_path = output_path.replace('.xlsx', '.json')
        # Use pandas to_json to reliably handle NaNs (converts to null)
        results_json = json.loads(df.to_json(orient='records', date_format='iso'))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
        
        if chain_validation:
            await send_progress(job_id, 99, "Search complete. Auto-starting Deep Validation...")
            
            # Prepare paths for validation chaining
            # Input for validator is the output of search
            val_input = output_path
            val_output = str(Path(output_path).parent / generate_output_filename("Retry", "Validated"))
            
            # Map columns for validator
            df_search = pd.read_excel(val_input)
            if 'Latitude' in df_search.columns:
                df_search['LAT'] = df_search['Latitude']
                df_search['LONG'] = df_search['Longitude']
            df_search.to_excel(val_input, index=False)
            
            # Run validator (reuse job_id for progress continuity)
            jobs[job_id]['script_type'] = 'validate' # Update type so UI knows? Or keep 'search'?
            # Actually, UI might expect 'completed' msg. We delay it.
            
            final_json = await run_validator_script(val_input, val_output, job_id)
            jobs[job_id]["output_file"] = val_output # Update final output
            return final_json
        
        await send_progress(job_id, 100, "Google Maps search complete!")
        return json_path
        
    except Exception as e:
        await send_progress(job_id, -1, f"Error: {str(e)}")
        raise e

# ============ Endpoints ============

@app.get("/")
async def root():
    return {"message": "GeoValidator API", "version": "1.0.0"}

@app.post("/run/{script_type}")
async def run_script(script_type: str, file: UploadFile = File(...)):
    """
    Upload an Excel file and run a script.
    script_type: 'validate' or 'search'
    """
    if script_type not in ['validate', 'search']:
        raise HTTPException(status_code=400, detail="Invalid script_type. Use 'validate' or 'search'.")
    
    # Job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    original_filename = file.filename or "upload"
    input_filename = f"{job_id}_{original_filename}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Output path with Better Naming
    output_filename = generate_output_filename(original_filename, script_type.capitalize())
    output_path = OUTPUT_DIR / output_filename
    
    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "script_type": script_type,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "result_json": None,
        "error": None
    }
    save_jobs()
    
    return {"job_id": job_id, "message": f"Job queued. Connect to WebSocket /ws/{job_id} for progress."}

@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    ws_connections[job_id] = websocket
    
    try:
        if job_id not in jobs:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            await websocket.close()
            return
        
        job = jobs[job_id]
        job["status"] = "running"
        
        # Send initial status
        await websocket.send_json({"type": "started", "job_id": job_id})
        
        # Run the appropriate script
        try:
            if job["script_type"] == "validate":
                result_path = await run_validator_script(
                    job["input_file"], 
                    job["output_file"], 
                    job_id
                )
            elif job["script_type"] == "retry_merge":
                result_path = await run_retry_merge_wrapper(
                    job["input_file"],
                    job["merge_base_file"],
                    job["output_file"],
                    job_id,
                    chain_validation=job.get("chain_validation", False)
                )
            else:
                result_path = await run_search_script(
                    job["input_file"],
                    job["output_file"],
                    job_id,
                    chain_validation=job.get("chain_validation", False)
                )
            
            job["status"] = "completed"
            job["result_json"] = result_path
            save_jobs()
            
            # Send completion message WITHOUT large payload
            await websocket.send_json({
                "type": "completed",
                # "results": results, # Removed to prevent WS size limit issues
                "download_url": f"/download/{job_id}",
                "results_url": f"/job/{job_id}/results"
            })
            
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            save_jobs()
            await websocket.send_json({"type": "error", "message": str(e)})
        
        # Keep connection open for a bit to ensure client receives final message
        await asyncio.sleep(1)
        
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in ws_connections:
            del ws_connections[job_id]

@app.post("/job/{job_id}/validate")
async def validate_job_results_deep(job_id: str, background_tasks: BackgroundTasks):
    """
    Trigger full Deep Validation on Search Results.
    Uses 'Latitude'/'Longitude' from search as 'LAT'/'LONG' for Validator.
    Returns a new job_id.
    """
    import fastapi
    import pandas as pd
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    old_job = jobs[job_id]
    input_source = old_job["output_file"]
    
    if not os.path.exists(input_source):
        raise HTTPException(status_code=404, detail="Result file not found")

    # Generate new job ID for the validation process
    new_job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare Input File (Map Cols)
    try:
        df = pd.read_excel(input_source)
        # Force map Search Cols -> Validator Cols
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df['LAT'] = df['Latitude']
            df['LONG'] = df['Longitude']
        
        # Save temp input
        temp_input = UPLOAD_DIR / f"{new_job_id}_deepval_input.xlsx"
        df.to_excel(temp_input, index=False)
        
        # Define output
        output_filename = f"output_validation_{timestamp}.xlsx"
        output_path = OUTPUT_DIR / output_filename
        
        # Register Job
        jobs[new_job_id] = {
            "status": "queued",
            "progress": 0,
            "script_type": "validate",
            "input_file": str(temp_input),
            "output_file": str(output_path),
            "result_json": None,
            "error": None
        }
        
        # Launch in background
        background_tasks.add_task(run_validator_script, str(temp_input), str(output_path), new_job_id)
        
        return {"new_job_id": new_job_id, "message": "Deep validation started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing validation: {str(e)}")

@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Signal job to cancel."""
    if job_id in jobs:
        jobs[job_id]["cancelled"] = True
        return {"status": "cancelled"}
    return {"status": "not_found"}

@app.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the JSON results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if not job["result_json"] or not Path(job["result_json"]).exists():
        raise HTTPException(status_code=404, detail="Results not available")
        
    return FileResponse(
        path=job["result_json"],
        media_type="application/json"
    )

@app.get("/download/template")
async def download_template():
    """Download the template spreadsheet."""
    # Using the path provided by the user
    template_path = Path("Planilha_MODELO.xlsx")
    if not template_path.exists():
        if not template_path.exists():
             template_path = Path("/Users/gousero/Documents/DEV/Validator Script/Planilha_MODELO.xlsx")
    
    if not template_path.exists():
         raise HTTPException(status_code=404, detail="Template file not found")
         
    return FileResponse(
        path=template_path,
        filename="Modelo_GeoLoc.xlsx", 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the output Excel file for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    output_path = Path(job["output_file"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        filename=f"resultado_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs and their status."""
    return {
        job_id: {
            "status": job["status"],
            "script_type": job["script_type"],
            "progress": job.get("progress", 0)
        }
        for job_id, job in jobs.items()
    }

@app.get("/files")
async def list_files():
    """List output files."""
    files = []
    for f in OUTPUT_DIR.glob("*.xlsx"):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
            "url": f"/download_file/{f.name}"
        })
    # Sort by modified desc
    files.sort(key=lambda x: x["modified"], reverse=True)
    return files

@app.get("/download_file/{filename}")
async def download_file_direct(filename: str):
    """Download specific file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


async def run_retry_merge_wrapper(
    subset_input_path: str,
    full_original_output_path: str,
    final_output_path: str,
    job_id: str,
    chain_validation: bool
):
    """
    Wrapper to run search/validate on SUSET of data, then MERGE back into FULL original data.
    Requires '_ORIGINAL_INDEX' column in subset_input_path.
    """
    import pandas as pd
    
    # 1. Run Search on Subset
    # We use a temporary output for the subset result
    subset_output_path = str(Path(subset_input_path).with_name(f"subset_result_{job_id}.xlsx"))
    
    # We rely on run_search_script to do the heavy lifting
    # It updates job progress.
    await run_search_script(subset_input_path, subset_output_path, job_id, chain_validation=chain_validation)
    
    # 2. Merge Logic
    await send_progress(job_id, 99, "Merging results...")
    
    try:
        # Load datasets
        df_full = pd.read_excel(full_original_output_path)
        df_subset = pd.read_excel(subset_output_path)
        
        # Ensure index column exists
        if '_ORIGINAL_INDEX' not in df_subset.columns:
            raise Exception("Retry subset missing '_ORIGINAL_INDEX'")
            
        # Update Full DF with new values
        # We iterate over subset and update df_full at index
        # We need to map columns from subset to full
        cols_to_update = [c for c in df_subset.columns if c != '_ORIGINAL_INDEX']
        
        for _, row in df_subset.iterrows():
            idx = int(row['_ORIGINAL_INDEX'])
            if idx < len(df_full):
                for col in cols_to_update:
                    # Update value
                    df_full.at[idx, col] = row[col]
        
        # 3. Save Final Full Output
        df_full.to_excel(final_output_path, index=False)
        
        # 4. Update Job with Final Output
        jobs[job_id]["output_file"] = final_output_path
        
        # 5. Save Final JSON
        json_path = final_output_path.replace('.xlsx', '.json')
        results_json = json.loads(df_full.to_json(orient='records', date_format='iso'))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
            
        jobs[job_id]["result_json"] = json_path
        save_jobs()
        
        return json_path

    except Exception as e:
        raise Exception(f"Merge failed: {e}")

@app.post("/job/{job_id}/retry_invalid")
async def retry_invalid_rows(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry logic (Smart Merge):
    1. Filter INVALID rows from original job.
    2. Create subset with _ORIGINAL_INDEX.
    3. Run Wrapper (Search -> Merge -> Save Full).
    """
    import pandas as pd
    
    if job_id not in jobs:
         raise HTTPException(status_code=404, detail="Job not found")
    
    old_job = jobs[job_id]
    output_file = old_job["output_file"]
    
    if not os.path.exists(output_file):
        # Fallback resolution
        fallback_path = OUTPUT_DIR / Path(output_file).name
        if fallback_path.exists():
            output_file = str(fallback_path)
            jobs[job_id]["output_file"] = output_file
            save_jobs()
        else:
             raise HTTPException(status_code=404, detail=f"Output file missing: {output_file}")
        
    df = pd.read_excel(output_file)
    
    # Filter Invalid
    if 'VALIDATION_STATUS' in df.columns:
        invalid_mask = df['VALIDATION_STATUS'] == 'INVALID'
    else:
        invalid_mask = pd.Series([True] * len(df)) # Retry all if no status
    
    invalid_df = df[invalid_mask].copy()
    
    if invalid_df.empty:
        raise HTTPException(status_code=400, detail="No INVALID rows found to retry.")
    
    # Add Index for Merging
    invalid_df['_ORIGINAL_INDEX'] = invalid_df.index
    
    # Create new job
    new_job_id = str(uuid.uuid4())[:8]
    
    # Save Subset Input
    input_stem = Path(output_file).stem
    input_path = UPLOAD_DIR / f"{new_job_id}_retry_{input_stem}.xlsx"
    invalid_df.to_excel(input_path, index=False)
    
    # Final Output Path (Full Dataset)
    output_filename = generate_output_filename(f"Retry_{input_stem}", "Merged")
    output_path = OUTPUT_DIR / output_filename
    
    jobs[new_job_id] = {
        "status": "queued",
        "progress": 0,
        "script_type": "retry_merge", # Use new type
        "chain_validation": True, # ENABLE CHAINING
        "input_file": str(input_path),
        "output_file": str(output_path),
        "merge_base_file": str(output_file), # Original file to merge into
        "result_json": None,
        "error": None
    }
    save_jobs()
    
    # Do NOT start background task here. WS leads execution.
    # background_tasks.add_task(...) REMOVED
    
    return {"new_job_id": new_job_id, "message": "Retry job started (Search + Auto-Validate)"}

@app.post("/job/{job_id}/retry_pending")
async def retry_pending_rows(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry logic (Smart Merge) for Pending:
    """
    import pandas as pd
    
    if job_id not in jobs:
         raise HTTPException(status_code=404, detail="Job not found")
    
    old_job = jobs[job_id]
    output_file = old_job["output_file"]
    
    if not os.path.exists(output_file):
        # Fallback
        fallback_path = OUTPUT_DIR / Path(output_file).name
        if fallback_path.exists():
            output_file = str(fallback_path)
            jobs[job_id]["output_file"] = output_file
            save_jobs()
        else:
            raise HTTPException(status_code=404, detail=f"Output file missing: {output_file}")
        
    df = pd.read_excel(output_file)
    
    pending_mask = pd.Series([False] * len(df))
    if 'Latitude' in df.columns:
        pending_mask = df['Latitude'].isna() | df['Longitude'].isna()
    elif 'LAT' in df.columns:
        pending_mask = df['LAT'].isna() | df['LONG'].isna()
    
    pending_df = df[pending_mask].copy()
    
    if pending_df.empty:
        raise HTTPException(status_code=400, detail="No PENDING rows found to retry.")
    
    # Add Index
    pending_df['_ORIGINAL_INDEX'] = pending_df.index
    
    # Create new job
    new_job_id = str(uuid.uuid4())[:8]
    
    input_stem = Path(output_file).stem
    input_path = UPLOAD_DIR / f"{new_job_id}_retry_pending_{input_stem}.xlsx"
    pending_df.to_excel(input_path, index=False)
    
    output_filename = generate_output_filename(f"RetryPending_{input_stem}", "Merged")
    output_path = OUTPUT_DIR / output_filename
    
    jobs[new_job_id] = {
        "status": "queued",
        "progress": 0,
        "script_type": "retry_merge", 
        "chain_validation": False, 
        "input_file": str(input_path),
        "output_file": str(output_path),
        "merge_base_file": str(output_file), # Original file
        "result_json": None,
        "error": None
    }
    save_jobs()
    
    # Do NOT start background task. WS leads execution.
    
    return {"new_job_id": new_job_id, "message": "Retry Pending job started (Smart Merge)"}

# ============ Run ============
if __name__ == "__main__":
    import uvicorn
    print("Starting GeoValidator API on http://localhost:8000")
    print("Frontend should connect to this backend for script execution.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
