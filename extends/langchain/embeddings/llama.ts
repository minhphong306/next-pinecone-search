import axios from 'axios';
import { Embeddings, EmbeddingsParams } from 'langchain/embeddings/base';
import { CreateEmbeddingRequestInput } from 'openai';
import { chunkArray } from '../util/chunk';

export interface LlamaEmbeddingsParams {
  /**
   * The maximum number of documents to embed in a single request. This is
   * limited by the OpenAI API to a maximum of 2048.
   */
  batchSize?: number;

  /**
   * Whether to strip new lines from the input text. This is recommended by
   * OpenAI, but may not be suitable for all use cases.
   */
  stripNewLines?: boolean;
}

export class LlamaEmbeddings
  extends Embeddings
  implements LlamaEmbeddingsParams
{
  batchSize = 512;
  stripNewLines = true;

  constructor(
    fields?: EmbeddingsParams & {
      stripNewLines?: boolean;
      batchSize?: number;
    }
  ) {
    const fieldsWithDefaults = { maxConcurrency: 2, ...fields };

    super(fieldsWithDefaults);

    this.stripNewLines = fields?.stripNewLines ?? this.stripNewLines;
    this.batchSize = fields?.batchSize ?? this.batchSize;
  }

  /**
   * Method to generate an embedding for a single document. Calls the
   * embeddingWithRetry method with the document as the input.
   * @param text Document to generate an embedding for.
   * @returns Promise that resolves to an embedding for the document.
   */
  async embedQuery(text: string): Promise<number[]> {
    const { data } = await this.embeddingWithRetry({
      input: this.stripNewLines ? text.replace(/\n/g, ' ') : text,
    });
    return data.data[0].embedding;
  }

  /**
   * Method to generate embeddings for an array of documents. Splits the
   * documents into batches and makes requests to the OpenAI API to generate
   * embeddings.
   * @param texts Array of documents to generate embeddings for.
   * @returns Promise that resolves to a 2D array of embeddings for each document.
   */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    const batches = chunkArray(
      this.stripNewLines ? texts.map((t) => t.replace(/\n/g, ' ')) : texts,
      this.batchSize
    );

    const batchRequests = batches.map((batch) =>
      this.embeddingWithRetry({
        input: batch,
      })
    );
    const batchResponses = await Promise.all(batchRequests);

    const embeddings: number[][] = [];
    for (let i = 0; i < batchResponses.length; i += 1) {
      const batch = batches[i];
      const { data: batchResponse } = batchResponses[i];
      for (let j = 0; j < batch.length; j += 1) {
        embeddings.push(batchResponse.data[j].embedding);
      }
    }
    return embeddings;
  }

  async embeddingWithRetry(request: { input: CreateEmbeddingRequestInput }) {
    return this.caller.call(
      axios.post<{
        object: string;
        model: string;
        usage: {
          prompt_tokens: number;
          total_tokens: number;
        };
        data: {
          object: string;
          embedding: number[];
        }[];
      }>,
      `${process.env.LLAMA_SERVICE_ORIGIN}/v1/embeddings`,
      request
    );
  }
}
