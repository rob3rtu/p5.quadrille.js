[Huggingface Reference](https://huggingface.co/blog/ngxson/make-your-own-rag)

## Try again with a more basic approach documented by hugging face in the link above.

This time I used simple data structures and 'hand made' functions. This implementation uses a simple list for the vector database. The embedding is made using the [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) model and the LLM used to generate the response is [`llama3`](https://ollama.com/library/llama3), both from ollama. Both models are running locally.

The chunking strategy used is to split the js code in chunks containing a whole function and its JSDoc definition(as you can see in the RAGs response below). I find this to be the best strategy in this case as each function keeps its whole logic.

To find the best chunks for in the retrieval phase I used the `cosine_similarity` function. May not be the best, but for now it seems to work ok.

### Update 1

I've added the .md docs into the RAGs knowledge. These docs are splited by md header(#). Using the current approach the code generates a total of 626 chunks(170 js and 456 md). Now that I've added a second category of information, in order to keep a balance between them in the retrieval stage I've decided to store them separate and get the model the best matches from both categories.

### Next steps from here

- ~~Include the .md docs into the RAG knowledge (not it only have the quadrille.js)~~
- Research better approach for the VectorDB
- Research a better alternative to cossine similarity function
- Research different models for embedding and LLM
- Improve demo experience: make it feel more like a chat by listening for the user prompt in a loop

## RAG output for the following prompt: "How can I invert filled and empty cells?"

Retrieved knowledge:

(similarity: 0.65) \*\*

- Inverts filled/empty status in-place.
- - Filled cells become empty (cleared).
- - Empty cells get filled with `target`.
- @param {\*} target Value to place into previously-empty cells
- @returns {Quadrille} this
  \*/
  not(target) {
  this.visit(({ row, col, value }) =>
  this.constructor.isFilled(value)
  ? this.clear(row, col)
  : this.fill(row, col, target)
  );
  return this;
  }

(similarity: 0.62) // Instance methods

/\*\*

- Checks whether the cell at (row, col) is empty.
- @param {number} row
- @param {number} col
- @returns {boolean}
  \*/
  isEmpty(row, col) {
  return this.constructor.isEmpty(this.read(row, col));
  }

(similarity: 0.61) /\*\*

- Randomly clears or fills cells in the quadrille.
- - If `value` is `null`, clears `times` filled cells.
- - If `value` is not `null`, fills `times` empty cells with `value`.
- Note: For deterministic behavior, call `randomSeed(seed)` explicitly before this method.
- @param {number} times - Number of cells to modify.
- @param {\*} [value=null] - Value to fill, or `null` to clear cells.
- @returns {Quadrille} The modified quadrille (for chaining).
  \*/
  rand(times, value = null) {
  times = this.\_p.abs(times);
  const isFilling = this.constructor.isFilled(value);
  const max = isFilling ? this.size - this.order : this.order;
  times = this.\_p.min(times, max);
  let count = 0;
  while (count < times) {
  const index = this.\_p.int(this.\_p.random(this.size));
  const { row, col } = this.\_fromIndex(index);
  const shouldChange = isFilling ? this.isEmpty(row, col) : this.isFilled(row, col);
  if (shouldChange) {
  isFilling ? this.fill(row, col, value) : this.clear(row, col);
  count++;
  }
  }
  return this;
  }

---

**LLM response**:

You can invert filled and empty cells using the `not(target)` method. This method inverts the filled/empty status in-place.

Here's an example:

```javascript
quadrille.not("X");
```

In this case, any cell that is currently marked as 'X' (filled) will become empty, and any cell that was previously empty will be marked as 'X'.
