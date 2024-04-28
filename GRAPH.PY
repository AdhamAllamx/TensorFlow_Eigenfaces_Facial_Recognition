# Attempt to correct issues with graph rendering and generating the PDF
from graphviz import Digraph

# Create a new directed graph
dot = Digraph()

# Adding nodes with labels for the research methods
dot.node('A', 'AI Facial Recognition on Low-Power Image Processing')
dot.node('B', 'Eigenfaces Method')
dot.node('C', 'Convolutional Neural Networks (CNN)')

# Adding nodes for the specific studies
dot.node('D', 'A Face Recognition System Based on Eigenfaces Method\n(Müge Çarıkçı, Figen Özen, 2012)')
dot.node('E', 'MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition\n(Yandong Guo et al., Microsoft Research)')

# Adding edges
dot.edge('A', 'B', label='Uses PCA')
dot.edge('A', 'C', label='Uses deep learning')
dot.edge('B', 'D', label='Study Example')
dot.edge('C', 'E', label='Study Example')

# Generate the PDF in the correct output path
output_path = "Chapter 2 AI-Based Facial Recognition Using Low Power Microprocessors"
dot.render(output_path, format='pdf', cleanup=True)

# Provide the path for download
output_path + ".pdf"
