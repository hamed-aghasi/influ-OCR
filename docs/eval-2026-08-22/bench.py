import base64, json, sys, time
sys.path.insert(0, '/Users/hamed/Desktop/influ-OCR/instagram_analyzer_app')
import os
os.environ.setdefault('SECRET_KEY', 'x'*32)
from processing.ocr_processor import EXTRACTION_PROMPT, RESPONSE_FORMAT, FrameResults
import requests

KEY = [l.split('=',1)[1].strip() for l in open('/Users/hamed/Desktop/influ-OCR/.env') if l.startswith('OPENROUTER_API_KEY=')][0]
FRAMES = sorted(f for f in os.listdir('frames') if f.endswith('.jpg'))
MODELS = ['qwen/qwen3-vl-235b-a22b-instruct']
PRICE = {'google/gemini-3.7-flash': (0.375, 1.875), 'qwen/qwen3.8-max': (2.0, 6.0), 'qwen/qwen3-vl-8b-instruct': (0.117, 0.455), 'qwen/qwen3-vl-235b-a22b-instruct': (0.21, 1.9)}

def run_model(model):
    out, cost = {}, 0.0
    for start in range(0, len(FRAMES), 10):
        batch = FRAMES[start:start+10]
        content = [{"type":"text","text":EXTRACTION_PROMPT}]
        for i, name in enumerate(batch):
            b64 = base64.b64encode(open(f'frames/{name}','rb').read()).decode()
            content.append({"type":"text","text":f"\nFrame {i}: {name}"})
            content.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        body = {"model":model,"messages":[{"role":"user","content":content}],
                "temperature":0.1,"max_tokens":8000,"response_format":RESPONSE_FORMAT}
        for attempt in range(3):
            try:
                r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                    headers={"Authorization":f"Bearer {KEY}"}, json=body, timeout=300)
                if r.status_code != 200:
                    print(f'  {model} batch {start//10}: HTTP {r.status_code} {r.text[:120]}', flush=True)
                    time.sleep(5); continue
                data = r.json()
                u = data.get('usage', {})
                pi, po = PRICE[model]
                cost += u.get('prompt_tokens',0)/1e6*pi + u.get('completion_tokens',0)/1e6*po
                frames = FrameResults.model_validate_json(data['choices'][0]['message']['content']).frames
                for fr in frames:
                    if 0 <= fr.frame_index < len(batch) and fr.metrics:
                        out[batch[fr.frame_index]] = {k:v for k,v in fr.metrics.model_dump().items() if v is not None}
                break
            except Exception as e:
                print(f'  {model} batch {start//10} attempt {attempt}: {type(e).__name__} {e}', flush=True)
                time.sleep(5)
    return out, cost

for model in MODELS:
    print(f'=== {model} ===', flush=True)
    t0 = time.time()
    results, cost = run_model(model)
    slug = model.split('/')[-1]
    json.dump(results, open(f'bench_{slug}.json','w'), indent=1)
    print(f'  frames with metrics: {len(results)}  cost: ${cost:.4f}  time: {time.time()-t0:.0f}s', flush=True)
print('BENCH DONE')
