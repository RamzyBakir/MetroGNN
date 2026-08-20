import asyncio
import unittest

from mcp_server.server import mcp


class MCPContractTests(unittest.TestCase):
    def test_tool_schemas_expose_ranges_and_enums(self):
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        query_schema = tools["query_papers"].input_schema["properties"]
        self.assertEqual(query_schema["top_k"]["minimum"], 1)
        self.assertEqual(query_schema["top_k"]["maximum"], 500)

        inference = tools["run_inference"].input_schema["properties"]
        self.assertEqual(
            inference["experience_level"]["enum"],
            ["beginner", "intermediate", "advanced"],
        )
        self.assertEqual(inference["sort_by"]["enum"], ["influence", "citations", "year"])

        training = tools["train_pipeline"].input_schema["properties"]
        self.assertEqual(training["epochs"]["minimum"], 1)
        self.assertEqual(training["epochs"]["maximum"], 2000)
        self.assertIn("force_topics", training)


if __name__ == "__main__":
    unittest.main()
