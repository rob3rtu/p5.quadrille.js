This first approach uses `LangChain`, which is a python library with multiple classes which can be used to build a RAG, like Chroma, GenericLoader, LanguageParser.

Reasons why I dont like this implementation and I will drop it:

- it is a blackbox implementation, which gives me less control
- it depends too much on langchain: if they make an update it may broke all I have
