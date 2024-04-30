import asyncio
import websockets

async def get_bnb_price():
    async with websockets.connect('wss://stream.binance.com:9443/ws/bnbusdt@aggTrade') as ws:
        while True:
            data = await ws.recv()
            print(data)

asyncio.get_event_loop().run_until_complete(get_bnb_price())
