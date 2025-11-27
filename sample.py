import requests
import json

payload = {
    "code": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None\n\n# Create nodes\nnode1 = Node(15)\nnode2 = Node(3)\nnode3 = Node(17)\nnode4 = Node(90)\n\n# Link nodes\nnode1.next = node2\nnode2.next = node3\nnode3.next = node4  \n\nhead = node1  # Head points to the first node\n\n# Traverse and print the linked list\ncurrent = head\nwhile current:\n    print(current.data, end=\" -> \")\n    current = current.next\nprint(\"None\")",
    "language": "python",
    "timeout": 10,
}

response = requests.post("http://localhost:8000/visualize/trace", json=payload)


output_filename = "trace_output.json"
with open(output_filename, "w") as json_file:
    json.dump(response.json(), json_file, indent=2)

