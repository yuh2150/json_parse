import time
import logging
from flask import Flask, request, jsonify
import llama_cpp
import queue
import threading
import requests
import json
import re
from typing import Any, List, Optional, Text, Dict
from geo_coding import GeoCodingAPI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
llm = llama_cpp.Llama(
    model_path="models/qwen2.5-1.5b-instructv6-tuningv6-q4_k_m.gguf",
    # flash_attn=True,
    # chat_format="chatml",
)
# llm = Llama.from_pretrained(
#     repo_id="yuh0512/SmolLM2-360M-Instruct-tuningv6-Q4_K_M-GGUF"
#     filename=""
# )
pick_up_result = None
destination_result = None

request_queue = queue.Queue()
response_dict = {}
def extract_value(match: Dict[Text, Any]) -> Dict[Text, Any]:
    
    if match["value"].get("type") == "interval":
        value = {
            "to": match["value"].get("to", {}).get("value"),
            "from": match["value"].get("from", {}).get("value"),
        }
    else:
        value = match["value"].get("value")

    return value
def extract_flight_code(text):
    pattern = r'[A-Z]{2}\d{1,4}'
    matches = re.findall(pattern, text)
    return matches

def convert_duckling_format_to_rasa(
    matches: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    extracted = []

    for match in matches:
        value = extract_value(match)
        grain = match['value'].get('from', {}).get('grain') or match['value'].get('grain') or match['value'].get('to', {}).get('grain') 
    
        if grain != 'year':
            entity = {
                "text": match.get("body", match.get("text", None)),
                "value": value,
                "additional_info": match["value"],
                "entity": match["dim"],
            }

            extracted.append(entity)

    return extracted
def filter_entities_by_dimensions(extracted: list, dimensions: list) -> list:
    if dimensions:
        return [entity for entity in extracted if entity.get('entity') in dimensions]
    return extracted
def getData_for_duckling(text, dims):
    url = 'http://localhost:8000/parse'
    data = {
        'locale': 'en_US',
        'text': text,
        'dims': dims,
        'tz': "Asia/Ho_Chi_Minh"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        json_response = response.json()
        # value = json_response[0]['value']['value']
        return json_response
    else:
        return f"Error: {response.status_code}"
# Worker to process requests from the queue
def worker():
    global pick_up_result, destination_result
    while True:
        request_id, messages = request_queue.get()
        try:
            logger.info(f"Processing request ID: {request_id}")
            print(messages)
            for message in messages:
                result = llm.create_chat_completion(
                    messages=[
                        { "role": "system",
                            "content": """
Task: Extract structured booking details from the provided text. Carefully analyze the context to determine each field's value.
Fields to Extract:
    name: The person's name, if mentioned.
    pickup location: The pickup location 
    destination location: The destination location
    passengers: The number of passengers. If not mentioned, set to null.
Extraction Rules:
    If a field is explicitly mentioned, extract its value.
    If a field is missing, set it to null.
    If only one address is given, determine whether it is a pickup location or destination location based on context.
    Extract passengers separately. If not specified, set passengers to null.
"""
                            },
                        {"role": "user", "content": message},
                    ],
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "nullable": True},
                                "from": {"type": "string", "nullable": True},
                                "to": {"type": "string", "nullable": True},
                                "passengers": {"type": "number", "nullable": True},
                            },"required": ["name","from", "to", "passengers"],
                        },
                    },
                    temperature=0.3,
                    top_k=50,
                    top_p=0.8
                )
                
                response_data= result.get("choices", [{}])[0].get("message", {}).get("content", {})
                response_data = str(response_data)
                response_data = json.loads(response_data)
                dimensions = ["time","email","phone-number"]
                data = getData_for_duckling(message,dimensions)
                flight_code = extract_flight_code(message)
                all_extracted = convert_duckling_format_to_rasa(data)
                extracted = filter_entities_by_dimensions(all_extracted, dimensions)
                # extracted =  json.loads(extracted)
                # extracted = extracted.json()
                time_value = next(
                    (val['value'] for item in extracted 
                    if item.get('entity') == 'time' 
                    for val in item.get('additional_info', {}).get('values', []) 
                    if val.get('grain') == 'minute' or val.get('grain') == 'hour' or val.get('grain') == 'second'or val.get('grain') == 'day'),
                    "null"
                )
                pickup_location = ""
                destination_location = ""
                geoCodingAPI = GeoCodingAPI()
        
                geoCoding_pickup = geoCodingAPI.get_geocoding(response_data.get("from"))
                if geoCoding_pickup["status"] == "OK" :
                    pick_up_result = geoCoding_pickup
                    pickup_location= geoCoding_pickup['results'][0]['formatted_address']
                else:
                    pickup_location= ""
                geoCoding_destination = geoCodingAPI.get_geocoding(response_data.get("to"))
                if geoCoding_destination["status"] == "OK" :
                    destination_result = geoCoding_destination
                    destination_location= geoCoding_destination['results'][0]['formatted_address']
                else:
                    destination_location= ""
                print(response_data)
                response_content = {
                    "name" : response_data.get("name"),
                    "pickup_location": pickup_location,
                    "destination_location": destination_location,
                    "passengers": response_data.get("passengers"),
                    "flight_code": flight_code[0] if flight_code and flight_code[0] is not None else None,
                    "email": next((item['value'] for item in extracted if item.get('entity') == 'email'), "null"),
                    "phone-number": next((item['value'] for item in extracted if item.get('entity') == 'phone-number'), "null"),
                    "pickup-time": time_value,
                }
                response_dict[request_id] = {"status": "done", "response": response_content}
        except Exception as e:
            logger.error(f"Error during inference for request ID {request_id}: {e}")
            response_dict[request_id] = {"status": "error", "response": str(e)}
        finally:
            request_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

@app.route('/api/booking', methods=['POST'])
def chat():
    global pick_up_result, destination_result
    input_data = request.json
    request_id = str(time.time())
    if not input_data or 'messages' not in input_data:
        return jsonify({'error': 'Invalid input'}), 400
    messages = input_data.get('messages', [])
    if not messages or not isinstance(messages, list):
        return jsonify({'error': 'Invalid input, messages must be a non-empty list'}), 400
    request_queue.put((request_id, messages))
    queue_contents = list(request_queue.queue)  # This gets the items in the queue
    print({"queue_contents": queue_contents}), 
    list_responses = []
    for message in messages:   
        # request_queue.put((request_id, message))
        response_dict[request_id] = {"status": "pending", "response": None}
        # Wait for the worker to process the message
        while response_dict[request_id]["status"] == "pending":
            time.sleep(0.0001)

        if response_dict[request_id]["status"] == "done":
            list_responses.append({
            "response":response_dict[request_id]["response"],
            "pick_up_result": pick_up_result,
            "destination_result": destination_result})
            pick_up_result = None  
            destination_result = None 
        else:
            list_responses.append({"error": response_dict[request_id]["response"]})
        
    response = {
        'responses': list_responses
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, threaded=True,debug=True)
