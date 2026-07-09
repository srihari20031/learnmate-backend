NOTE_PROMPT = """You are a technical educator writing a focused study note for a developer learning {target_tech} who already knows: {known_stack}.

Topic: {topic}

From their known stack above, SILENTLY pick the ONE or TWO most RELEVANT technologies to anchor the comparison for THIS topic (e.g. a web-framework topic → compare to Express; a database topic → compare to MongoDB). Do NOT list their whole stack in any heading or sentence — that's noise.

Structure your response like this:
## What is {topic}?
(brief, clear explanation)

## How you'd approach it with what you already know
(Use the ONE most relevant technology they know, with a short code example. BUT: if {target_tech} is a DIFFERENT category from anything they know — e.g. an infrastructure/DevOps tool vs app development — do NOT force a fake comparison; instead relate this topic to the relevant workflow they already know.)

## How it works in {target_tech}
(explanation with a short code example)

## Key Differences
(concise bullet points)
"""
