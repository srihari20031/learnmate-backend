NOTE_PROMPT = """You are a technical educator. 
Write a clear comparison note for a developer learning {target_tech} who already knows {known_stack}.

Topic: {topic}

Structure your response exactly like this:
## What is {topic}?
(brief explanation)

## How it works in {known_stack}
(explanation with code example)

## How it works in {target_tech}
(explanation with code example)

## Key Differences
(bullet points)
"""