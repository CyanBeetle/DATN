import { NextResponse } from 'next/server';
import xml2js from 'xml2js';

export async function GET() {
    try {
        const rssUrl = 'https://vnexpress.net/rss/tin-moi-nhat.rss';
        const res = await fetch(rssUrl);
        const xmlText = await res.text();

        const parser = new xml2js.Parser();
        const result = await parser.parseStringPromise(xmlText);
        const items = result.rss.channel[0].item;

        return NextResponse.json(items);
    } catch (error) {
        console.error('Error fetching news:', error);
        return NextResponse.json({ error: 'Failed to fetch news' }, { status: 500 });
    }
}
