import dotenv from "dotenv";
dotenv.config();
const OPENAI_API_KEY = process.env.OPENAI_KEY;

// Document loader: Essentially scrapes the website and returns the text
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);
const data = await loader.load();
// Document splitter: WE split the document into semantically meaningful chunks
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});

const splitDocs = await textSplitter.splitDocuments(data);
// We then have to store the documents in a vector store. We use the LLM to convert the documents into vectors.
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const embeddings = new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY });

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
// We then perform a similarity search in the vector DB to find the most relevant documents
const relevantDocs = await vectorStore.similaritySearch(
  "What is task decomposition?"
);

console.log(relevantDocs.length);
// Now we pass on the relevant documents to the QA model
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: OPENAI_API_KEY,
});
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.call({
  query: "What is task decomposition?",
});
console.log(response);

/*
  {
    text: 'Task decomposition refers to the process of breaking down a larger task into smaller, more manageable subgoals. By decomposing a task, it becomes easier for an agent or system to handle complex tasks efficiently. Task decomposition can be done through various methods such as using prompting or task-specific instructions, or through human inputs. It helps in planning and organizing the steps required to complete a task effectively.'
  }
*/
