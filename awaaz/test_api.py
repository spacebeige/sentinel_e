#!/usr/bin/env python3
"""
AWAAZ FastAPI - Test & Demo Script
Tests all three layers of the voice processing pipeline
"""

import asyncio
import json
import time
import requests
from pathlib import Path
from typing import Dict, Optional

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
HEALTH_CHECK_URL = "http://localhost:8000/health"
PIPELINE_INFO_URL = "http://localhost:8000/api/v1/pipeline/background-info"

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_section(text: str):
    """Print formatted section."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}→ {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def health_check() -> bool:
    """Perform API health check."""
    print_section("Health Check")
    
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server is {data['status'].upper()}")
            print_info(f"Uptime: {data['uptime_seconds']:.1f} seconds")
            
            # Check components
            for comp, status in data['components'].items():
                if status['status'] == 'initialized':
                    print_success(f"  {comp}: {status['message']}")
                else:
                    print_error(f"  {comp}: {status['message']}")
            
            return data['status'] == 'healthy'
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to API: {e}")
        return False

def get_pipeline_info():
    """Get pipeline background information."""
    print_section("Pipeline Background Information")
    
    try:
        response = requests.get(PIPELINE_INFO_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_info(f"Pipeline: {data['pipeline_name']} v{data['version']}")
            print_info(f"Description: {data['description']}")
            
            print(f"\n{Colors.OKBLUE}Layers:{Colors.ENDC}")
            for layer in data['layers']:
                print(f"  Layer {layer['layer_number']}: {layer['name']}")
                print(f"    Description: {layer['description']}")
                for provider in layer['providers']:
                    status_color = Colors.OKGREEN if provider['status'] == 'initialized' else Colors.FAIL
                    print(f"    {status_color}Provider: {provider['name']} ({provider['status']}){Colors.ENDC}")
            
            print(f"\n{Colors.OKBLUE}Supported Languages:{Colors.ENDC}")
            langs = data['layers'][0]['supported_languages']
            print(f"  {', '.join(langs)}")
    except Exception as e:
        print_error(f"Failed to get pipeline info: {e}")

def upload_voice_file(file_path: str) -> Optional[str]:
    """Upload voice file for transcription."""
    print_section("Layer [1/3] - Upload Voice File")
    
    if not Path(file_path).exists():
        print_error(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{API_BASE_URL}/transcription/upload",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print_success(f"File uploaded successfully")
            print_info(f"  Job ID: {job_id}")
            print_info(f"  Filename: {data['filename']}")
            print_info(f"  Size: {data['size_bytes']} bytes")
            print_info(f"  Status URL: {data['status_url']}")
            return job_id
        else:
            print_error(f"Upload failed: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Upload error: {e}")
        return None

def start_transcription(job_id: str, language: Optional[str] = None) -> bool:
    """Start transcription process."""
    print_section("Layer [1/3] - Start Transcription")
    
    try:
        params = {'job_id': job_id}
        if language:
            params['language'] = language
        
        response = requests.post(
            f"{API_BASE_URL}/transcription/process-async",
            params=params,
            timeout=5
        )
        
        if response.status_code == 200:
            print_success("Transcription started")
            print_info(f"  Job ID: {job_id}")
            return True
        else:
            print_error(f"Failed to start transcription: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def wait_for_transcription(job_id: str, max_wait: int = 60) -> Optional[Dict]:
    """Poll transcription status until completion."""
    print_section("Layer [1/3] - Waiting for Transcription")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{API_BASE_URL}/transcription/status/{job_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'completed':
                    print_success("Transcription completed!")
                    print_info(f"  Language: {data['detected_language']}")
                    print_info(f"  Confidence: {data['confidence']:.2%}")
                    print_info(f"  Duration: {data['execution_time_ms']:.0f}ms")
                    print_info(f"  Text: {data['transcribed_text']}")
                    return data
                
                elif data['status'] == 'failed':
                    print_error(f"Transcription failed: {data['error_message']}")
                    return None
                
                else:
                    print_info(f"Status: {data['status']}...")
            
            time.sleep(2)
        
        except Exception as e:
            print_error(f"Error checking status: {e}")
            time.sleep(2)
    
    print_error(f"Transcription timeout after {max_wait} seconds")
    return None

def process_with_ai(text: str, language: str, job_id: Optional[str] = None) -> Optional[str]:
    """Send to AI for processing."""
    print_section("Layer [2/3] - AI Processing")
    
    try:
        params = {
            'text': text,
            'language': language
        }
        if job_id:
            params['job_id'] = job_id
        
        response = requests.post(
            f"{API_BASE_URL}/ai-processing/process-async",
            params=params,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            ai_job_id = data['job_id']
            print_success("AI processing started")
            print_info(f"  Job ID: {ai_job_id}")
            return ai_job_id
        else:
            print_error(f"AI processing failed: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def wait_for_ai_processing(job_id: str, max_wait: int = 60) -> Optional[Dict]:
    """Poll AI processing status until completion."""
    print_section("Layer [2/3] - Waiting for AI Response")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{API_BASE_URL}/ai-processing/status/{job_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'completed':
                    print_success("AI processing completed!")
                    print_info(f"  Model: {data['processing_model']}")
                    print_info(f"  Duration: {data['execution_time_ms']:.0f}ms")
                    print_info(f"  Response: {data['ai_response']}")
                    return data
                
                elif data['status'] == 'failed':
                    print_error(f"AI processing failed: {data['error_message']}")
                    return None
                
                else:
                    print_info(f"Status: {data['status']}...")
            
            time.sleep(2)
        
        except Exception as e:
            print_error(f"Error checking status: {e}")
            time.sleep(2)
    
    print_error(f"AI processing timeout after {max_wait} seconds")
    return None

def start_tts_synthesis(text: str, language: str, speaker: str = "ritu") -> Optional[str]:
    """Start TTS synthesis."""
    print_section("Layer [3/3] - TTS Synthesis")
    
    try:
        params = {
            'text': text,
            'language': language,
            'speaker': speaker
        }
        
        response = requests.post(
            f"{API_BASE_URL}/tts/synthesize-async",
            params=params,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            tts_job_id = data['job_id']
            print_success("TTS synthesis started")
            print_info(f"  Job ID: {tts_job_id}")
            print_info(f"  Speaker: {speaker}")
            return tts_job_id
        else:
            print_error(f"TTS synthesis failed: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def wait_for_tts(job_id: str, max_wait: int = 60) -> Optional[Dict]:
    """Poll TTS status until completion."""
    print_section("Layer [3/3] - Waiting for Audio Synthesis")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{API_BASE_URL}/tts/status/{job_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'completed':
                    print_success("TTS synthesis completed!")
                    print_info(f"  Provider: {data['tts_provider']}")
                    print_info(f"  Duration: {data['audio_duration_s']:.1f}s")
                    print_info(f"  Execution time: {data['execution_time_ms']:.0f}ms")
                    print_info(f"  Path: {data['audio_path']}")
                    return data
                
                elif data['status'] == 'failed':
                    print_error(f"TTS synthesis failed: {data['error_message']}")
                    return None
                
                else:
                    print_info(f"Status: {data['status']}...")
            
            time.sleep(2)
        
        except Exception as e:
            print_error(f"Error checking status: {e}")
            time.sleep(2)
    
    print_error(f"TTS timeout after {max_wait} seconds")
    return None

def download_audio(job_id: str, output_path: str) -> bool:
    """Download synthesized audio."""
    print_section("Downloading Audio")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/tts/download/{job_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print_success(f"Audio downloaded to: {output_path}")
            return True
        else:
            print_error(f"Download failed: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def get_pipeline_details(job_id: str) -> Optional[Dict]:
    """Get complete pipeline details."""
    print_section("Pipeline Execution Details")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/pipeline/background-details/{job_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Failed to get details: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def demo_full_pipeline(audio_file: str, language: str = "hi"):
    """Run complete E2E pipeline demo."""
    print_header("AWAAZ FASTAPI - FULL PIPELINE DEMO")
    
    # Step 1: Health check
    if not health_check():
        print_error("API is not healthy. Please start the server first.")
        return
    
    # Step 2: Get pipeline info
    get_pipeline_info()
    
    # Step 3: Upload audio
    job_id = upload_voice_file(audio_file)
    if not job_id:
        return
    
    # Step 4: Transcription
    if not start_transcription(job_id, language):
        return
    
    transcription_data = wait_for_transcription(job_id)
    if not transcription_data:
        return
    
    text = transcription_data['transcribed_text']
    detected_lang = transcription_data['detected_language']
    
    # Step 5: AI Processing
    ai_job_id = process_with_ai(text, detected_lang, job_id)
    if not ai_job_id:
        return
    
    ai_data = wait_for_ai_processing(ai_job_id)
    if not ai_data:
        return
    
    ai_response = ai_data['ai_response']
    
    # Step 6: TTS Synthesis
    tts_job_id = start_tts_synthesis(ai_response, detected_lang)
    if not tts_job_id:
        return
    
    tts_data = wait_for_tts(tts_job_id)
    if not tts_data:
        return
    
    # Step 7: Download audio
    output_file = f"output_{job_id[:8]}.wav"
    download_audio(tts_job_id, output_file)
    
    # Step 8: Pipeline summary
    print_header("PIPELINE EXECUTION COMPLETE ✅")
    
    pipeline_details = get_pipeline_details(job_id)
    if pipeline_details:
        total_time = pipeline_details['total_execution_ms']
        print_info(f"Total execution time: {total_time:.0f}ms ({total_time/1000:.1f}s)")
        print_success(f"Input: {text}")
        print_success(f"Output: {ai_response}")
        print_success(f"Audio: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else "hi"
        demo_full_pipeline(audio_file, language)
    else:
        print(f"{Colors.BOLD}Usage:{Colors.ENDC}")
        print(f"  python3 test_api.py <audio_file> [language_code]")
        print(f"\n{Colors.BOLD}Example:{Colors.ENDC}")
        print(f"  python3 test_api.py recording.wav hi")
        print(f"  python3 test_api.py speech.mp3 pa")
        print(f"\n{Colors.BOLD}Supported languages:{Colors.ENDC}")
        print(f"  hi, pa, ta, te, kn, ml, bn, or, gu, mr, en, ur, as, ne, sa")
