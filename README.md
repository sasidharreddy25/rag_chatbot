# rag_chatbot

Stepup your openai keys in config.py

Then open testing.ipynb and install all requirements. Run each cell sequentially. Add all your queries to the `queries` list and specify the PDF file path as well.

### Improve Accuracy
- **Prompt Engineering**: Crafting precise prompts and employing techniques such as Chain-of-Thought (CoT) and ReACT, among others.
- **Chunking**: Utilizing advanced chunking strategies like semantic chunking, agentic chunking, and markdown chunking etc.
- **Retrieval**: Employing a combination of dense retrievers and cross-encoder-based retrievers for re-ranking, such as ColBERT and similar methods.
- **Other Techniques**: Utilize additional techniques such as multi-hop querying for handling multiple queries within a single query, and use GraphRAG to identify better relationships in the document.
- **Hyperparameters**: Using different hyperparameters like temperature, top-k, top-p etc. and k values in vectordb.

### Code Modular
Following the S.O.L.I.D Principles more efficiently:
- **S** - Single Responsibility Principle
- **O** - Open/Closed Principle
- **L** - Liskov Substitution Principle
- **I** - Interface Segregation Principle
- **D** - Dependency Inversion Principle

Doing better error handling and logging in the code.

Doing better documentation of the code. Use a separate class to initialize all keys.
