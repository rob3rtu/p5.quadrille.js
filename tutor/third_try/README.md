[Huggingface Reference](https://huggingface.co/blog/ngxson/make-your-own-rag)

## Try again with a more basic approach documented by hugging face in the link above.

This time I used simple data structures and 'hand made' functions. This implementation uses a simple list for the vector database. The embedding is made using the [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) model and the LLM used to generate the response is [`llama3`](https://ollama.com/library/llama3), both from ollama. Both models are running locally.

The chunking strategy used is to split the js code in chunks containing a whole function and its JSDoc definition(as you can see in the RAGs response below). I find this to be the best strategy in this case as each function keeps its whole logic.

To find the best chunks for in the retrieval phase I used the `cosine_similarity` function. May not be the best, but for now it seems to work ok.

### Update 1

I've added the .md docs into the RAGs knowledge. These docs are splited by md header(#). Using the current approach the code generates a total of 626 chunks(170 js and 456 md). Now that I've added a second category of information, in order to keep a balance between them in the retrieval stage I've decided to store them separate and get the model the best matches from both categories.

### Update 2

I've updated the chat to run in a loop and included some more files in the context: the addon.js and the main README.md of the repo. Running a simple query as "How to use this library?" I found some issues with the current code:

- The best chunk retrieved contains only the header of the function 'visit' and has a similarity of just 56%, which I find pretty low.
- The RAGs response misses some implementation details, as it tells me to initialize a qudrille like `let quadrille = new Quadrille()`, instead of using the `createQuadrille()` method added to p5. It also see some functions headers but tells me to write the implementation for those, instead of telling me how to use them because they are already implemented.

Some posible thinks to look into that may cause the issues:

- poor .md and .js files indexing
- poor similarity function (cosine similarity)
- weak models (llm - llama3, embedding - nomic-embed-text')
- too little context retrieved (3 js and 3 md)

### Update 3

I've implemented a `multi_query` function to improve the retrieval, but in order to use it I need to fix the files indexing. I print the results in multi_query.txt: they look ok, I will check later if I can improve it.
Now the conversation history is saved and the model can use it. It is printed in the debug file I've added for analysis, conversation_history.txt.
Next step will be to retrieve more chunks based on the multi query, filter out duplicates and improve the files indexing, as it seems to be some problems not just on .md files but on .js too.

### Update 4

I've adapted the `retrieve` function to work on multi queries. Now that I get multiple chunks found by multiple queries, I improved the ranking system. Instead on relying only on the cossine similarity, I've implemented `Reciprocal Rank Fusion (RRF)`. After getting the best chunks for each query using cossine similarity, I use RRF to rank the chunks considering their similarity with all queries. So if a chunk is in top 3 for multiple queries it will get a higher score comparing to a chunk that is 1st only for 1 query. I can't say now how good it is because I still need to fix the indexing step, which I will do next.

### Update 5

Improve md indexing: instead of splitting the files by header now I encode the whole file, as they are pretty small and should fit the model's context window. Will check that later

#### Update 5.1

With the next md indexing I got a small issue. The biggest chunk is about 4k tokens while the nomic-embed-text is limited to a 2k tokens context window. I will look for a model with bigger context window or a new indexing method for the md files

#### Update 5.2

I tried [`qwen3-embedding:0.6b`](https://ollama.com/library/qwen3-embedding) for embedding as it has a 32k context window, but the retrieval results are awful: for a very simple query like `How to install it?` it can't find any information and returns `I cannot find that information in the provided documentation`. Will investigate

#### Update 5.3

I've got the RAG to a stable version now, and it is decent again!! I changed the embedding model to [`Embeddinggemma`](https://ollama.com/library/embeddinggemma) from Google. I changed the .md parsing to split the docs by header and also improved the js parsing, as it was choping the functions. The `multi query` step was also improved with new instructions when there is no conversation history

## Update 6

I managed to improve the retrieval by adjusting the prompt for the multi query step in order to include a `Step Back` query and adjusting the params regarding the number of chunk to be used. While the first prompt result is better than before, a follow up question may be less accurate. Considering this issue, next I will focus on improving the usage of the conversation history and already retrieved chunks. CUrrently the model only sees the last 6 messages(3 questions from user and 3 answers from model), but the chunks related to them are forgotten, and they may not be retrieved again for the follow up which will make the next answers worse.

## Update 7

Adjust the prompts to improve the usage of the conversation history. This includes making sure the model uses the full code examples he gets from the retrieved document to be able to use them later from the history.

### TO DO

- ~~Include the .md docs into the RAG knowledge (not it only have the quadrille.js)~~
- ~~Add current conversation to the context as it goes on~~
- Research better approach for the VectorDB
- Research a better alternative to cossine similarity function
- Research different models for embedding and LLM
- ~~Improve demo experience: make it feel more like a chat by listening for the user prompt in a loop~~

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
