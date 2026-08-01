"""
Main entrypoint — prototype CLI.

WHY THIS FILE IS INTENTIONALLY THIN:
All conversation logic lives in core/orchestrator.py. This file's only
job is to provide a way to run the agent from the command line during
prototype development. At MVP, a FastAPI app will replace this as the
primary entrypoint -- and it will import handle_message from the same
orchestrator, with zero changes to the pipeline itself.

HOW TO RUN:
    python -m app.main

PREREQUISITES:
    1. python -m app.rag.build_embeddings   (run once to build the index)
    2. HF_TOKEN set in your .env file       (required for response generation)
"""

import uuid

from app.logging_config import setup_logging
from app.schemas.models import CustomerMessage
from app.core.orchestrator import handle_message


def main() -> None:
    # Configure logging before anything else.
    setup_logging()

    print("=" * 50)
    print("  StoreFlow AI — Prototype Sales Agent")
    print("  Type 'quit' to exit.")
    print("=" * 50)

    conversation_id = str(uuid.uuid4())
    customer_id = "prototype_user"

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        message = CustomerMessage(
            conversation_id=conversation_id,
            customer_id=customer_id,
            text=user_input,
            channel="prototype",
        )

        print("\n[Processing...]\n")
        reply = handle_message(message)

        print(f"Agent : {reply.reply_text}")
        print(f"\n  emotion : {reply.emotion.label.value} ({reply.emotion.confidence:.0%})")
        print(f"  intent  : {reply.intent.label.value} ({reply.intent.confidence:.0%})")

        if reply.recommendations:
            names = [r.name for r in reply.recommendations]
            print(f"  recs    : {', '.join(names)}")


if __name__ == "__main__":
    main()