// Graph.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <list>
#include <queue>

template<class T> class Node {
public:
	T data;
	Node* next;
	Node(T val, Node* nxt = nullptr) {
		data = val;
		next = nxt;
	}
};

template<class T> class List {
private:
	Node<T>* first;
public:
	List() {
		first = nullptr;
	}
	List(const List& lst) : List() {
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
	}
	~List() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
	}
	List<T>& operator=(const List<T> lst) {
		first = nullptr;
		Node<T>* current = nullptr;
		for (iterator it = lst.begin(); it != end(); ++it) {
			current = insert_after(it->data, current);
		}
		return *this;
	}
	Node<T>* getFirst() {
		return first;
	}
	T front() {
		return first->data;
	}
	bool empty() {
		return first == nullptr;
	}
	void clear() {
		iterator it = begin();
		while (it != end()) {
			iterator nit(it->next);
			delete* it;
			it = nit;
		}
		first = nullptr;
	}
	Node<T>* insert_after(T data, Node<T>* n) {
		if (n == nullptr) return push_front(data);
		Node<T>* pnn = new Node<T>(data, n->next);
		n->next = pnn;
		return pnn;
	}
	void erase_after(Node<T>* n) {
		if (n == nullptr) pop_front();
		Node<T>* rem = n->next;
		n->next = rem->next;
		delete rem;
	}
	Node<T>* push_front(T data) {
		Node<T>* pnn = new Node<T>(data, first);
		first = pnn;
		return pnn;
	}
	void pop_front() {
		Node<T>* rem = first;
		first = rem->next;
		delete rem;
	}
	class iterator {
	private:
		Node<T>* current;
	public:
		iterator() {
			current = first;
		}
		iterator(Node<T>* pnode) {
			current = pnode;
		}
		iterator(const iterator& it) {
			current = it.current;
		}
		Node<T>*& operator*() {
			return current;
		}
		Node<T>* operator->() {
			return current;
		}
		iterator operator++() {
			current = current->next;
			return current;
		}
		bool operator==(iterator it) {
			return it.current == current;
		}
		bool operator!=(iterator it) {
			return !(it == *this);
		}
	};
	iterator begin() const {
		iterator it(first);
		return it;
	}
	iterator end() const {
		iterator it(nullptr);
		return it;
	}
};

template<class T> struct Pair {
	unsigned int first;
	T second;
	Pair() {

	}
	Pair(unsigned int n1, T n2) {
		first = n1;
		second = n2;
	}
	Pair(const Pair<T>& p) {
		first = p.first;
		second = p.second;
	}
	Pair& operator=(const Pair p) {
		first = p.first;
		second = p.second;
		return *this;
	}
};

template<class T> struct Vertex {
	Pair<T> data;
	List<Vertex<T>*> adjacencyList;
	Vertex(Pair<T> d) {
		data = d;
	}
	Vertex(Pair<T> d, List<Vertex<T>*> al) {
		data = d;
		adjacencyList = al;
	}
	Vertex(const Vertex<T>& vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
	}
	Vertex<T>& operator=(const Vertex<T> vx) {
		data = vx.data;
		adjacencyList = vx.adjacencyList;
		return *this;
	}
};

template<class T> class Graph {
private:
	void BFS(Vertex<T>* current, std::vector<bool>& markedVertices,
		std::vector<Vertex<T>*>& path, std::vector<int>& distances, std::vector<Vertex<T>*>& predecessors, std::queue<Vertex<T>*>& q) {
		q.push(current);
		while (!(q.empty())) {
			Vertex<T>* u = q.front();
			q.pop();
			for (typename List<Vertex<T>*>::iterator it = u->adjacencyList.begin(); it != u->adjacencyList.end(); ++it) {
				if (!(markedVertices[index((*it)->data)])) {
					markedVertices[index((*it)->data)] = true;
					path.push_back((*it)->data);
					distances[index((*it)->data)] = distances[index(u)] + 1;
					predecessors[index((*it)->data)] = u;
					q.push((*it)->data);
				}
			}
		}
	}
	void DFS(Vertex<T>* current, std::vector<bool>& markedVertices, std::vector<Vertex<T>*>& path) {
		markedVertices[index(current)] = true;
		path.push_back(current);
		for (typename List<Vertex<T>*>::iterator it = current->adjacencyList.begin(); it != current->adjacencyList.end(); ++it) {
			bool marked = false;
			if (markedVertices[index((*it)->data)]) continue;
			DFS((*it)->data, markedVertices, path);
		}
		return;
	}
public:
	std::vector<Vertex<T>> vertices;
	/*Graph(Vertex<T> r) {

		std::vector<Vertex<T>*> markedVertices;
		root = new Vertex<T>(r.data);
		markedVertices.push_back(root);
		createVertex(root, markedVertices, r);


	}
	*/
	Graph(std::vector<Vertex<T>> v) {
		vertices = v;
	}
	/*
	~Graph() {
		delete[] root;
	}
	*/
	/*
	void createVertex(Vertex<T>* pvx, std::vector<Vertex<T>*>& markedVertices, Vertex<T> vx) {
		for (typename List<Vertex<T>*>::iterator it = vx.adjacencyList.begin(); it != vx.adjacencyList.end(); ++it) {
			bool marked = false;
			for (int i = 0; i < markedVertices.size(); i++)
				if ((*it)->data->data.first == markedVertices[i]->data.first) {
					marked = true;
					pvx->adjacencyList.push_front(markedVertices[i]);
				}
			if (marked) continue;
			Vertex<T>* newVertex = pvx->adjacencyList.push_front(new Vertex<T>((*it)->data->data))->data;
			markedVertices.push_back((*it)->data);
			createVertex(newVertex, markedVertices, (*it)->data->data);
			newVertex->adjacencyList.push_front(pvx);
		}
	}

	void deleteBranch(Vertex<T>* current, std::vector<Vertex<T>*>& markedVertices) {
		for (typename List<Vertex<T>*>::iterator it = current->adjacencyList.begin(); it != current->adjacencyList.end(); ++it) {
			bool marked = false;
			for (int i = 0; i < markedVertices.size(); i++)
				if ((*it)->data == markedVertices[i]) {
					marked = true;
				}
			if (marked) continue;
			markedVertices.push_back((*it)->data);
			deleteBranch((*it)->data, markedVertices);
		}
	}
	*/
	int index(Vertex<T>* pvx) {
		int index = pvx - &(vertices[0]);
		return index;
	}

	std::vector<Vertex<T>*> BFS(int startingVertex, std::vector<int>* d = nullptr, std::vector<Vertex<T>*>* pred = nullptr) {
		std::vector<bool> markedVertices(vertices.size());
		markedVertices[startingVertex] = true;
		std::vector<Vertex<T>*> path = {};
		path.push_back(&(vertices[startingVertex]));
		std::vector<int> distances(vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			if (i == startingVertex) distances[i] = 0;
			else distances[i] = -1;
		}
		std::vector<Vertex<T>*> predecessors(vertices.size());
		std::queue<Vertex<T>*> q;
		BFS(&(vertices[startingVertex]), markedVertices, path, distances, predecessors, q);
		if (d != nullptr)
			*d = distances;
		if (pred != nullptr)
			*pred = predecessors;
		return path;
	}

	std::vector<Vertex<T>*> DFS(int startingVertex = 0) {
		std::vector<bool> markedVertices(vertices.size());
		std::vector<Vertex<T>*> path = {};
		DFS(&(vertices[startingVertex]), markedVertices, path);
		for (int i = 0; i < vertices.size(); i++)
			if (!(markedVertices[i])) DFS(&(vertices[i]), markedVertices, path);
		return path;
	}
};


int main()
{
	std::vector<Vertex<int>> vertices;
	for (int i = 0; i <= 8; i++) {
		vertices.push_back(Vertex<int>(Pair<int>(i, i * 12), List<Vertex<int>*>()));
	}

	Graph<int> graph(vertices);
	graph.vertices[0].adjacencyList.push_front(&(graph.vertices[1])); //   8-0-7
	graph.vertices[0].adjacencyList.push_front(&(graph.vertices[7])); //   | | |
	graph.vertices[0].adjacencyList.push_front(&(graph.vertices[8])); //   3-1-2
	graph.vertices[1].adjacencyList.push_front(&(graph.vertices[0])); //   |\ /|
	graph.vertices[1].adjacencyList.push_front(&(graph.vertices[2])); //   5-6-4
	graph.vertices[1].adjacencyList.push_front(&(graph.vertices[3]));
	graph.vertices[2].adjacencyList.push_front(&(graph.vertices[1]));
	graph.vertices[2].adjacencyList.push_front(&(graph.vertices[4]));
	graph.vertices[2].adjacencyList.push_front(&(graph.vertices[6]));
	graph.vertices[2].adjacencyList.push_front(&(graph.vertices[7]));
	graph.vertices[3].adjacencyList.push_front(&(graph.vertices[1]));
	graph.vertices[3].adjacencyList.push_front(&(graph.vertices[5]));
	graph.vertices[3].adjacencyList.push_front(&(graph.vertices[6]));
	graph.vertices[3].adjacencyList.push_front(&(graph.vertices[8]));
	graph.vertices[4].adjacencyList.push_front(&(graph.vertices[2]));
	graph.vertices[4].adjacencyList.push_front(&(graph.vertices[6]));
	graph.vertices[5].adjacencyList.push_front(&(graph.vertices[3]));
	graph.vertices[5].adjacencyList.push_front(&(graph.vertices[6]));
	graph.vertices[6].adjacencyList.push_front(&(graph.vertices[2]));
	graph.vertices[6].adjacencyList.push_front(&(graph.vertices[3]));
	graph.vertices[6].adjacencyList.push_front(&(graph.vertices[4]));
	graph.vertices[6].adjacencyList.push_front(&(graph.vertices[5]));
	graph.vertices[7].adjacencyList.push_front(&(graph.vertices[0]));
	graph.vertices[7].adjacencyList.push_front(&(graph.vertices[2]));
	graph.vertices[8].adjacencyList.push_front(&(graph.vertices[0]));
	graph.vertices[8].adjacencyList.push_front(&(graph.vertices[3]));
	/*
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << graph.index(&(graph.vertices[i])) << " ";
		*/
	std::vector<Vertex<int>*> path = graph.DFS(0);
	for (int i = 0; i < path.size(); i++)
		std::cout << path[i]->data.first << " ";
	std::cout << std::endl << std::endl;
	std::vector<int> distances;
	std::vector<Vertex<int>*> predecessors;
	path = graph.BFS(0, &distances, &predecessors);
	for (int i = 0; i < path.size(); i++)
		std::cout << path[i]->data.first << " ";
	std::cout << std::endl;
	std::cout << std::endl;
	for (int i = 0; i < graph.vertices.size(); i++)
		std::cout << distances[i] << " ";
	std::cout << std::endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

/*
vertices[0].adjacencyList.push_front(&(vertices[2]));
	vertices[0].adjacencyList.push_front(&(vertices[3]));
	vertices[1].adjacencyList.push_front(&(vertices[2]));
	vertices[1].adjacencyList.push_front(&(vertices[3]));
	vertices[2].adjacencyList.push_front(&(vertices[1]));
	vertices[2].adjacencyList.push_front(&(vertices[4]));
	vertices[2].adjacencyList.push_front(&(vertices[0]));
	vertices[3].adjacencyList.push_front(&(vertices[1]));
	vertices[3].adjacencyList.push_front(&(vertices[0]));
	vertices[4].adjacencyList.push_front(&(vertices[2]));
	*/