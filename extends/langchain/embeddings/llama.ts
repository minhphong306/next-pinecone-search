import axios from 'axios';
import { Embeddings, EmbeddingsParams } from 'langchain/embeddings/base';

export class LlamaEmbeddings extends Embeddings {
  constructor(params?: EmbeddingsParams) {
    super(params || {});
  }

  async embedQuery(text: string) {
    const result = await this.embedWithRetry(text);

    return result.data[0].embedding;
  }

  async embedDocuments(documents: string[]) {
    const embeddings: number[][] = [];

    for (const document of documents) {
      const result = await this.embedWithRetry(document);

      embeddings.push(result.data[0].embedding);
    }

    return embeddings;
  }

  async embedWithRetry(input: string) {
    try {
      const result = await axios.post<{
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
      }>(`${process.env.LLAMA_SERVICE_ORIGIN}/v1/embeddings`, {
        input,
      });
      return result.data;
    } catch (error: any) {
      throw error;
    }
  }
}
