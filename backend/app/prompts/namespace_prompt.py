# def build_namespace_prompt(user_query: str, namespaces: list) -> str:
#     """
#     Build the system prompt for GenAI to select the most relevant namespace.

#     Args:
#         user_query (str): The user's query.
#         namespaces (list): List of dicts containing website metadata.

#     Returns:
#         str: Formatted system prompt.
#     """
#     context_blocks = []
#     for ns in namespaces:
#         block = f"""
#             Namespace: {ns.get('namespace')}
#             URL: {ns.get('url')}
#             Title: {ns.get('title')}
#             Description: {ns.get('description')}
#             Keywords: {ns.get('keywords')}
#             Main Heading: {ns.get('main_heading')}
#             """
#         context_blocks.append(block)

#     context = "\n".join(context_blocks)

#     system_prompt = f"""
#     You are an expert AI system that selects the correct namespace based on a user's query. 
#     You are given multiple website metadata entries (url, title, description, keywords, main heading, namespace).
#     Your task is to analyze the user query and match it with the MOST relevant namespace.

#     Rules:
#     - Return ONLY the namespace string.
#     - Do NOT include explanations or extra text.

#     Namespaces dataset:
#     {context}

#     User query: "{user_query}"
#     """
#     return system_prompt
