import sys

from agent import build_reader_agent, build_search_agent, critic_chain, writer_chain


def safe_print(value):
    try:
        print(value)
    except UnicodeEncodeError:
        text = str(value).encode("utf-8", errors="replace").decode("utf-8")
        print(text)


def run_research_pipeline(topic: str) -> str:
    state = {}

    if sys.stdout.encoding and "cp1252" in sys.stdout.encoding.lower():
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    safe_print("\n" + " =" * 50)
    safe_print("step 1 - search agent is working ...")
    safe_print("=" * 50)

    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
    except Exception as exc:
        state["search_results"] = f"Search failed: {exc}"

    safe_print("\n search result ")
    safe_print(state["search_results"])

    safe_print("\n" + " =" * 50)
    safe_print("step 2 - Reader agent is Scraping the doc ...")
    safe_print("=" * 50)

    try:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
    except Exception as exc:
        state["scraped_content"] = f"Reader failed: {exc}"

    safe_print("\nscraped content: \n")
    safe_print(state["scraped_content"])

    safe_print("\n" + " =" * 50)
    safe_print("step 3 - Writer is drafting the report ...")

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    try:
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
    except Exception as exc:
        state["report"] = f"Writer failed: {exc}"
        safe_print("\n Writer error: ")
        safe_print(state["report"])

    safe_print("\n Final Report\n")
    safe_print(state["report"])

    safe_print("\n" + " =" * 50)
    safe_print("step 4 - critic is reviewing the report ")
    safe_print("=" * 50)

    try:
        state["feedback"] = critic_chain.invoke({
            "report": state["report"]
        })
    except Exception as exc:
        state["feedback"] = f"Critic failed: {exc}"
        safe_print("\n Critic error: ")
        safe_print(state["feedback"])

    safe_print("\n critic report \n")
    safe_print(state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
