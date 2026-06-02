import json
from app.core.cache import redis_client

CURRICULUM_TTL = 60 * 60 * 24 * 7 
NOTE_TTL = 60 * 60 * 24 * 7
SESSION_TTL = 60 * 60 * 24

def get_cached_curriculum(known_stack: str, target_stack: str):
    key = f"curriculum:{known_stack.lower()}:{target_stack.lower()}"
    cached_data = redis_client.get(key)
    if cached_data:
        print(f"Cache HIT: {key}")
        return json.loads(cached_data)
    else:
        print(f"Cache MISS: {key}")
        return None
    
def set_cached_curriculum(known_stack: str, target_stack: str, topics: list):
    key = f"curriculum:{known_stack.lower()}:{target_stack.lower()}"
    print(f"Setting cache: {key} with TTL {CURRICULUM_TTL} seconds")
    redis_client.set(key, json.dumps(topics), ex=CURRICULUM_TTL)
    
def get_cached_note(topic: str, known_stack: str, target_stack: str):
    key = f"note:{topic.lower()}:{known_stack.lower()}:{target_stack.lower()}"
    cached_data = redis_client.get(key)
    if cached_data:
        print(f"Cache HIT: {key}")
        return json.loads(cached_data)
    else:
        print(f"Cache MISS: {key}")
        return None
    
def set_cached_note(topic: str, known_stack: str, target_stack: str, note: str):
    key = f"note:{topic.lower()}:{known_stack.lower()}:{target_stack.lower()}"
    print(f"Setting cache: {key} with TTL {NOTE_TTL} seconds")
    redis_client.set(key, json.dumps(note), ex=NOTE_TTL)