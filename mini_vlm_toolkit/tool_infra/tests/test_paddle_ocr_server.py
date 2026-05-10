from tools.ocr.tool import OCR_Tool         
from tools.ocr.process_tool_output import process_tool_output
                                        
result = OCR_Tool().execute(image_path="tests/data/test_ocr.png")                                                                                                                   
print(result["text"])  # raw Chinese, not \u escapes                                                                                                                                                            
                                                                                                                                                                                                                
agent_msg = process_tool_output(                                                                                                                                                                                
    {"ok": True, "tool_name": "OCR_Tool", "result": result}, prepared=None
)                                                                                                                                                                                                               
print(agent_msg["user_message"]["content"])