# Rubric-level judge interface

The grader receives one user query, one assistant response, and one
case-specific rubric criterion at a time.

```text
Your job is to evaluate whether the assistant's response meets the criteria of a specific rubric item.

# User Query
{user_message}

# Assistant Response
{response}

# Rubric Item to Evaluate
{rubric_item}

# Instructions
Evaluate whether the assistant's response meets the criteria described in the rubric item above.

Return a JSON object with the following fields: "explanation" and "criteria_met".
- "explanation": A concise explanation of why the response does or does not meet the criteria.
- "criteria_met": A boolean (true/false) indicating whether the response meets the criteria.

Important rules:
1. If a rubric item has multiple criteria or sentences, ALL must be met for criteria_met to be true. If any single criterion is not met, return false.
2. If a criterion uses cue words such as "such as", "for example", or "including", the response does not need to include every listed example; it only needs to address the general requirement adequately.
3. Judge factual medical accuracy. If the rubric specifies a drug, dosage, guideline, or medical fact, the response must be consistent with current medical knowledge.
4. If the rubric asks whether a condition is satisfied, evaluate strictly: the response must clearly demonstrate the required behavior.
```

## Output schema

```json
{
  "explanation": "Brief criterion-level rationale.",
  "criteria_met": true
}
```

The public metric code consumes rubric-level boolean decisions. It does not
depend on the judge's explanation text.

