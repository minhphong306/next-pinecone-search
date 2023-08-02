import { NextRequest, NextResponse} from "next/server";
import { PineconeClient} from "@pinecone-database/pinecone";
import { queryPineconeVectorStoreAndQueryLLM} from "@/utils";
import {indexName} from "@/config";

export async function POST(req: NextRequest) {
    const body = await req.json();
    const client = new PineconeClient();
    await client.init({
        // @ts-ignore
        apiKey: process.env.PINECONE_API_KEY,
        // @ts-ignore
        environment: process.env.PINECONE_ENVIRONMENT,
    })

    const text = await queryPineconeVectorStoreAndQueryLLM(client, indexName, body)
    return NextResponse.json({
        data: text
    })
}
