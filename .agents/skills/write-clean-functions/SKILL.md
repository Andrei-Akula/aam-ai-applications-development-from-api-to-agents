---
name: write-clean-functions
description: **REQUIRED** — When writing, creating, editing, or refactoring any function: you MUST complete the refinement process described here before the implementation is considered done. A working first draft is not sufficient. Apply the extraction and abstraction steps before presenting code to the user.
metadata: 
  keywords: coding, refactoring, best practices, clean code
---

# Important Principles

- Functions should be small
- Each function should lead you to the next in a compelling order
- All functions in a module should be at the same level of abstraction!!!

# Key points about writing functions

The key points about writing functions from "Clean Code" regarding the process:

1. First Draft Approach:
- Write the function first to make it work, don't worry about cleanliness initially
- It's okay if the first draft is long and complicated
- Get the logic working first

2. Refining Process:
- Break down long functions into smaller ones
- Choose clear names for the new functions
- Maintain consistent abstraction levels
- Extract duplicate code
- Reorganize and refactor until the code is clean

3. Key Steps in Refinement:
- Identify sections that can be extracted into separate functions
- Look for clues like comments (they often indicate where a new function should be)
- Ensure each function tells a story
- Keep refactoring until each function does just one thing

4. Signs You're Done:
- Can't extract any more meaningful functions
- Function names clearly describe what they do
- Functions are small and focused
- Code reads like well-written prose