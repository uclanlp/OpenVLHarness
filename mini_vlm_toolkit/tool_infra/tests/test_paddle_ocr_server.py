import argparse                                                                                                                                                                                               
import json                                                                         
import mimetypes                                                                                                                                                                                              
import os                                                                            
import sys                                                                          
                                                                                            
import requests                                                                      
from pathlib import Path

IMAGE_PATH = Path(__file__).resolve().parent / "data" / "test_ocr.png"
                                                                                            
DEFAULT_SERVER_URL = "http://127.0.0.1:8085"                                         
                                                                                    
                                                                                            
def parse_args():                                                                    
    p = argparse.ArgumentParser(description="Smoke test the OCR server.")           
    p.add_argument("--image_path", default=IMAGE_PATH, help="Local image path to upload.")                       
    p.add_argument(                                                                  
        "--server-url",                                                             
        default=os.getenv("OCR_SERVER_URL", DEFAULT_SERVER_URL),                           
        help=f"OCR server base URL (default: env OCR_SERVER_URL or {DEFAULT_SERVER_URL}).",
    )                                                                               
    p.add_argument(                                                                        
        "--timeout",                                                                       
        type=float,                                                                 
        default=180.0,                                                                 
        help="Request timeout in seconds (default: 180).",                            
    )                                                                               
    return p.parse_args()

def check_health(server_url):                                                                                                                                                                                 
    health_url = f"{server_url.rstrip('/')}/health"                                                  
    print(f"[health] GET {health_url}")                                                                                                                                                                       
    response = requests.get(health_url, timeout=10)                                        
    response.raise_for_status()                                                                                                                                                                               
    payload = response.json()                                                       
    print(f"[health] ok: {json.dumps(payload, indent=2)}")                                                                                                                                                    
    return payload                                                                                                                                                                                            
                                                                                    
                                                                                                                                                                                                                
def run_ocr(server_url, image_path, timeout):                                              
    if not os.path.isfile(image_path):                                              
        raise FileNotFoundError(f"image not found: {image_path}")                          
                                                                                            
    endpoint = f"{server_url.rstrip('/')}/v1/ocr"                                   
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"       
    print(f"[ocr] POST {endpoint}  image={image_path}  ({content_type})")                  
                                                                                    
    with open(image_path, "rb") as handle:                                                 
        files = {"image": (os.path.basename(image_path), handle, content_type)}            
        response = requests.post(endpoint, files=files, timeout=timeout)            
                                                                                            
    if response.status_code != 200:                                                        
        print(f"[ocr] FAIL {response.status_code}: {response.text}", file=sys.stderr)
        response.raise_for_status()                                                        
                                                                                            
    payload = response.json()                                                        
    text = payload.get("text", "")                                                         
    print(f"[ocr] ok  model={payload.get('model_id')}  device={payload.get('device')}")    
    print(f"[ocr] text length: {len(text)} chars")                                   
    print("---- transcribed text ----")                                                
    print(text)                                                                        
    print("---- end ----")                                                           
    return payload


def main():                                                                          
    args = parse_args()                                                                    
    check_health(args.server_url)                                                          
    run_ocr(args.server_url, args.image_path, args.timeout)                          
                                                                                            
                                                                                            
if __name__ == "__main__":                                                           
    main()