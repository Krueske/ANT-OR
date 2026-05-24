## Image-Text Relevance Evaluation Criteria

## I. Score Level Definitions

| Score | Level | Core Definition |
|---|---|---|
| **2 points** | Highly Relevant | The image directly and completely answers the query or intent. By viewing the image, the user can obtain a clear answer or useful information. |
| **1 point** | Weakly Relevant | The image has some connection to the query and provides partial information or related atmosphere, but it does not fully satisfy the search intent. |
| **0 points** | Irrelevant | The image has no obvious relation to the query and provides no valuable information. |

## II. Detailed Scoring Standards

### Criteria for 2 Points (Highly Relevant)

**All of the following conditions must be met:**

1. **Subject Match**: The image clearly shows the core subject of the query (person, object, scene, or concept).

2. **Semantic Completeness**: If the query contains constraints or modifiers (e.g., "red apple"), the subject in the image must satisfy those conditions.

3. **Intent Fulfillment**: The image can directly respond to the user’s question, need, or search intent.

**Typical cases:**

- Query: "Eiffel Tower" → The image clearly shows the Eiffel Tower.
- Query: "red sports car" → The image shows a red sports car; both the color and the vehicle type match.
- Query: "boiled egg" → The image shows either the process of boiling eggs or the finished boiled eggs.

### Criteria for 1 Point (Weakly Relevant)

**An image can be rated 1 point if it meets any one of the following conditions:**

| Weak Relevance Type | Criteria | Example |
|---|---|---|
| **Partial Match** | The image contains some elements of the query but does not express it completely. | Query: "sunset at the beach" → The image shows the beach but no sunset. |
| **Missing Constraints** | The main subject matches, but the query’s modifiers or conditions are not satisfied. | Query: "red apple" → The image shows a green apple. |
| **Atmosphere Match** | The image does not directly correspond to the literal meaning of the query, but conveys a related mood or feeling. | Query: "romantic dinner" → The image shows candlelight but no food. |
| **Indirect Association** | The image is only indirectly related to the query through logic or context. | Query: "weight loss" → The image shows gym equipment. |

### Criteria for 0 Points (Irrelevant)

**An image should be rated 0 points if it meets any one of the following conditions:**

| Irrelevance Type | Criteria | Example |
|---|---|---|
| **Completely Mismatched** | The image content is entirely different from the literal meaning of the query. | Query: "cat" → The image shows a car. |
| **Misleading** | The image may lead to an incorrect interpretation or contradict the query. | Query: "healthy diet" → The image shows junk food. |
| **Lack of Information** | The image is too abstract, blurry, or contains no recognizable elements. | Query: "Great Wall" → The image is a solid color. |
| **Unintelligible** | A normal user cannot establish any logical connection between the image and the query. | Query: "2+2" → The image shows a flower. |

## III. Ambiguous Queries (Directly Judged as Irrelevant)

### What Counts as an "Ambiguous Query"

An ambiguous query is one that lacks a clear referent, making it impossible to determine what kind of image the user wants. Such queries should be directly judged as irrelevant. Common types include:

| Type | Characteristics | Examples |
|---|---|---|
| **Vague Expression** | Colloquial or incomplete phrases | "Come take a look", "Just browsing" |
| **Question Form** | Interrogative sentences that require an informational answer | "How do I cancel my membership?", "My appointment registration" |
| **Permission Check** | Asking whether something is allowed or possible | "Can I apply?" |
| **Subjective Description** | Lacks a concrete, objective entity | "Is this thing useful?" |
