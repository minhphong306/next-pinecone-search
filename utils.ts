import {OpenAIEmbeddings} from 'langchain/embeddings/openai'
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import {OpenAI} from 'langchain/llms/openai'
import {loadQAStuffChain} from "langchain/chains";
import {Document} from "langchain/document";
import {indexName, timeout} from './config'
import {PineconeClient} from "@pinecone-database/pinecone";

export const createPineconeIndex = async (
    client: PineconeClient,
    indexName: string,
    vectorDimension: number,
) => {
    // 1. Initiate index existence check
    console.log(`Checking "${indexName}"...`)
    // 2. Get list of existing indexes
    const existingIndexes = await client.listIndexes()
    // 3. If index doesn't exist, create it
    if (!existingIndexes.includes(indexName)) {
        // 4. Log index creation initation
        console.log(`Creating "${indexName}"...`)
        // 5. Create index
        await client.createIndex({
            createRequest: {
                name: indexName,
                dimension: vectorDimension,
                metric: 'cosine'
            }
        })
        // 6. Log successful creation
        console.log(`Creating index... please wait for it to finish initializing.`)
        // 7. wait for index initialization
        await new Promise((resolve) => setTimeout(resolve, timeout))
    } else {
        // 8. Log if index already exists
        console.log(`"${indexName}" already exists.`);
    }
}

export const updatePinecone = async (
    client: PineconeClient,
    indexName: string,
    docs: Document[]
) => {
    // 1. Retrieve Pinecone index
    const index = client.Index((indexName))
    // 2. Log the retrieved index name
    console.log(`Pinecone index retrieved: ${indexName}`)
    for (const doc of docs) {
        console.log(`Processing document: ${doc.metadata.source}`);
        const txtPath = doc.metadata.source;
        const text = doc.pageContent;
        // 4. Create RecursiveCharacterTextSplitter instance
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
        })
        console.log('Splitting text into chunks...');
        // 5. Split text into chunks (documents)
        const chunks = await textSplitter.createDocuments([text]);
        console.log(`Text split into ${chunks.length} chunks`);
        console.log(`Calling OpenAI's Embedding endpoint documents with ${chunks.length} text chunks ...`)
        // 6. Create OpenAI embeddings for documents
        const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
            chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
        )
        console.log(`Creating ${chunks.length} vectors array with id, values, and metadata...`)

        // 7. Create and upsert vectors in batches of 100
        const batchSize = 100
        let batch: any = [];
        for (let idx = 0; idx < chunks.length; idx++) {
            const chunk = chunks[idx];
            const vector = {
                id: `${txtPath}_${idx}`,
                values: embeddingsArrays[idx],
                metadata: {
                    ...chunk.metadata,
                    loc: JSON.stringify(chunk.metadata.loc),
                    pageContent: chunk.pageContent,
                    txtPath: txtPath
                }
            }
            batch = [...batch, vector]
            // When batch is full, or it's the last item, upsert the vectors
            if (batch.length === batchSize || idx == chunks.length-1) {
                await index.upsert({
                    upsertRequest: {
                        vectors: batch,
                    }
                })

                // Empty the batch
                batch = [];
            }

        }
    }
}

export const queryPineconeVectorStoreAndQueryLLM = async (
    client: PineconeClient,
    indexName: string,
    question: string
) => {
    // 1. Start query process
    console.log('Querying Pinecone vector store...');
    // 2. Retrieve the Pinecone index
    const index = client.Index(indexName);
    // 3. Create query embedding
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question);
    // 4. Query Pinecone index and return top 10 matches
    let queryResponse = await index.query({
        queryRequest: {
            topK: 10,
            vector: queryEmbedding,
            includeMetadata: true,
            includeValues: true
        }
    })
    // 5. Log the number of matches
    console.log(`Found ${queryResponse.matches?.length} matches...`)
    console.log(`Got vectors:`)
    console.log(JSON.stringify(queryResponse.matches))
    // 6. Log the question being asked
    console.log(`Asking question: ${question}...`);
    if (queryResponse.matches?.length) {
        // 7. Create an OpenAPI  instance and load the QAStuffChain
        const llm = new OpenAI({});
        const chain = loadQAStuffChain(llm);
        // 8. Extract and concatenate page content from matched documents
        const concatenatedPageContent = queryResponse.matches.map((match) => {
            // @ts-ignore
            return match?.metadata?.pageContent;
        }).join(" ");
        const result = await chain.call({
            input_documents: [new Document({ pageContent: concatenatedPageContent})],
            question: `${question}, trả lời bằng thơ nhé`,
        })
        // 10. log the answer
        console.log(`Answer: ${result.text}`)
        return result.text;
    } else {
        // 11. Log that there is no matches, so GPT-3 will not be queried.
        console.log("There is no matches, so GPT-3 will not be queried.")
    }
}
