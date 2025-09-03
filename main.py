# suhpai-model/main.py

from supreme_karaka_parser import SupremeKarakaParser
from suhpai_core import SUHPaiCore
import sys
import asyncio

def main():
    print("Welcome to Supreme Ultra Hybrid Paninian AI (SUHPAI)!")
    print("Loading lightweight models with advanced enhancements...")
    
    try:
        karaka_parser = SupremeKarakaParser()
        ai = SUHPaiCore(karaka_parser)
        asyncio.run(ai.run_hypothesis_engine())  # Start hypothesis engine
    except Exception as e:
        print(f"Init failed: {e}. Ensure deps installed.")
        sys.exit(1)

    print("\nReady! Type 'quit' to exit.")
    print("\nExamples:")
    print(" - Who hit the ball?")
    print(" - He sent it from where?")
    print(" - Who is near Paris?")
    print(" - Add fact: Alice lives in Berlin at morning for income.")  # Extended syntax
    print(" - कौन गेंद मारा? (Who hit the ball?)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("SUHPAI: Bye!")
            break
        
        response = ai.process_query(user_input)
        print(f"SUHPAI: {response}")

if __name__ == "__main__":
    main()
