#include "GraphClass.h"
static const int UNREACHABLE = -1;
extern Environment env;

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))


void Graph::collectPaths(int source, int dest, std::vector<std::vector<int>>& paths, std::vector<int>* predecessors) {
	/*if (predecessors[dest].size() == 0) {
		predecessors.push_back({});
		return;
	}*/
	if (predecessors[dest].size() == 0 || predecessors[dest][0] == source) {
		paths.push_back({ dest });
		return;
	}
	int depth;
	std::stack<int> st, depths;
	int rememberedDepth = 0;
	std::vector<int> currPath = {  };
	st.push(dest);
	depths.push(1);
	while (!st.empty()) {
		int node = st.top();


		//if (node == source)
			//return;
		st.pop();
		depth = depths.top();
		depths.pop();
		for (int i = 0; i < rememberedDepth - depth + 1; i++)
			currPath.pop_back();
		if (node != source)
			currPath.push_back(node);
		for (int i = 0; i < predecessors[node].size(); i++) {
			if (predecessors[node][i] == source) {
				paths.push_back(currPath);
				continue;
			}

			//paths[paths.size() - 1].push_back(predecessors[node][i]);
			st.push(predecessors[node][i]);
			depths.push(depth + 1);
			//collectPaths(source, index(predecessors[dest][i]), paths, predecessors, depth + 1);
			//if(newPath) currPaths.pop_back();
		}
		rememberedDepth = depth;
	}
}
void Graph::brandesBFS(int startingVertex, std::vector<int> clusters, std::vector<std::vector<int>>& predecessors,
	std::vector<double>& deltas, std::vector<int>& sigmas, std::vector<double>& cbs/*, std::vector<double>* localDeltas = nullptr*/) {
	//std::vector<bool> markedVertices(size());
	//std::vector<std::vector<int>> predecessors(size());
	int currCluster = clusters[startingVertex];
	std::vector<std::vector<std::vector<int>>> paths(size());
	//markedVertices[startingVertex] = true;
	std::vector<int> path = {};
	path.push_back(startingVertex);
	std::vector<int> distances(size());
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::deque<int> q;
	std::stack<int> s;
	q.push_back(startingVertex);
	while (!q.empty()) { //
		int u = q.front();
		q.pop_front();
		s.push(u);
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (clusters[it] != currCluster) continue;
			if (distances[it] == UNREACHABLE) {
				//markedVertices[it] = true;
				//path.push_back(it);
				distances[it] = distances[u] + 1;
				//predecessors[it] = u;
				q.push_back(it);
			}
			if (distances[u] + 1 == distances[it]) {
				predecessors[it].push_back(u);
				sigmas[it] += sigmas[u];
			}
		}
	} //
	while (!s.empty()) {
		int w = s.top();
		s.pop();
		for (int i = 0; i < predecessors[w].size(); i++) {
			deltas[predecessors[w][i]] += double(sigmas[predecessors[w][i]]) / sigmas[w] * (1 + deltas[w]);
		}
		if (w != startingVertex) cbs[w] += deltas[w];
	}
}
void Graph::modifiedBrandesV1BFS(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors,
	double* deltas, int* sigmas, double* localDeltas, int* localSigmas, int* distances,
	std::vector<bool>& externalNodes) {
	//std::vector<bool> markedVertices(size());
	//std::vector<std::vector<int>> predecessors(size());
	int currCluster = clusters[startingVertex];
	//std::vector<std::vector<std::vector<int>>> paths(size());
	//markedVertices[startingVertex] = true;
	//std::vector<int> path = {};
	//path.push_back(&(vertices[startingVertex]));
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::deque<int> q;
	std::stack<int> s;
	q.push_back(startingVertex);
	while (!q.empty()) { //
		int u = q.front();
		q.pop_front();
		s.push(u);
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (clusters[it] != currCluster) continue;
			if (distances[it] == UNREACHABLE) {
				//markedVertices[it] = true;
				//path.push_back(it);
				distances[it] = distances[u] + 1;
				//predecessors[it] = u;
				q.push_back(it);
			}
			if (distances[u] + 1 == distances[it]) {
				predecessors[it].push_back(u);
				sigmas[it] += sigmas[u];
			}
		}
	} //
	while (!s.empty()) {
		int w = s.top();
		s.pop();
		for (int i = 0; i < predecessors[w].size(); i++) {
			deltas[predecessors[w][i]] += double(sigmas[predecessors[w][i]]) / sigmas[w] * (int(!externalNodes[w]) + deltas[w]);
		}
		if (w != startingVertex) {
			if (externalNodes[w]) continue;
			localDeltas[w] += deltas[w];
		}
		localSigmas[w] = sigmas[w];
	}
}
void Graph::modifiedBrandesV2BFS(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors,
	double* globalDeltas, int nClusters) {
	//std::vector<bool> markedVertices(size());
	//std::vector<std::vector<int>> predecessors(size());
	int currCluster = clusters[startingVertex];
	double** deltas = new double* [size()];
	//std::vector<std::vector<double>> deltas(size());
	for (int i = 0; i < size(); i++) {
		deltas[i] = new double[nClusters];
		for (int j = 0; j < nClusters; j++)
			deltas[i][j] = 0;
	}
	std::vector<int> sigmas(size());
	sigmas[startingVertex] = 1;
	for (int i = 0; i < nClusters; i++) {
		deltas[startingVertex][i] = 1;
	}
	std::vector<int> distances(size());
	//std::vector<std::vector<std::vector<int>>> paths(size());
	//markedVertices[startingVertex] = true;
	//std::vector<int> path = {};
	//path.push_back(&(vertices[startingVertex]));
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::deque<int> q;
	std::stack<int> s;
	q.push_back(startingVertex);
	while (!q.empty()) { //
		int u = q.front();
		q.pop_front();
		s.push(u);
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			//if (clusters[it] != currCluster && clusters[u] != currCluster) continue;
			if (distances[it] == UNREACHABLE) {
				//markedVertices[it] = true;
				//path.push_back(it);
				distances[it] = distances[u] + 1;
				//predecessors[it] = u;
				q.push_back(it);
			}
			if (distances[u] + 1 == distances[it]) {
				predecessors[it].push_back(u);
				sigmas[it] += sigmas[u];
			}
		}
	} //
	while (!s.empty()) {
		int w = s.top();
		s.pop();
		for (int i = 0; i < predecessors[w].size(); i++) {
			int v = predecessors[w][i];
			for (int j = 0; j < nClusters; j++) {
				deltas[v][j] += double(sigmas[v]) / sigmas[w] * (int(clusters[w] == j) + deltas[w][j]);
			}
		}
		if (w != startingVertex) {
			if (clusters[w] != currCluster) {
				globalDeltas[w] += 2 * deltas[w][clusters[w]];
				//globalDeltas[w] += deltas[w][clusters[w]];
				for (int j = 0; j < nClusters; j++) {
					if (j == clusters[w]) continue;
					else globalDeltas[w] += deltas[w][j];
				}
			}
			//localSigmas[w] += sigmas[w];
		}
	}
	//std::cout << globalDeltas[0] << std::endl;
	for (int i = 0; i < size(); i++) {
		delete[] deltas[i];
	}
	delete[] deltas;
}
void Graph::findBorderNodes(std::vector<int>& clusters, std::vector<bool>& res) {
	// std::vector<bool> res(size());
#pragma omp parallel for shared(res)
	for (int i = 0; i < size(); i++) {
		if (res[i]) continue;
		int currCluster = clusters[i];
		for (int j = row_index[i]; j < row_index[i + 1]; j++) {
			int it = adjacencies[j];
			if (clusters[it] != currCluster) {
				res[i] = true;
				break;
			}
		}
	}
}
void Graph::localBFSAllPaths(int startingVertex, std::vector<int>& clusters, std::vector<int>* predecessors) {
	//std::vector<bool> markedVertices(size());
	//std::vector<std::vector<int>> predecessors(size());
	int currCluster = clusters[startingVertex];
	//std::vector<std::vector<std::vector<int>>> paths(size());
	//markedVertices[startingVertex] = true;
	std::vector<int> path = {};
	path.push_back(startingVertex);
	std::vector<int> distances(size());
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::deque<int> q;
	q.push_back(startingVertex);
	while (!q.empty()) { //
		int u = q.front();
		q.pop_front();
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (clusters[it] != currCluster) continue;
			if (distances[it] == UNREACHABLE) {
				//markedVertices[it] = true;
				//path.push_back(it);
				distances[it] = distances[u] + 1;
				//predecessors[it] = u;
				q.push_back(it);
			}
			if (distances[u] + 1 == distances[it]) {
				predecessors[it].push_back(u);
			}
		}
	} //
}
 void Graph::buildHSN(std::vector<int>& clusters, std::vector<bool>& borderNodes, std::vector<bool>& hsn) {
	//std::vector<bool> hsn(size());
	std::vector<int>* predecessorVector = new std::vector<int>[size() * nThreads];
	std::vector<std::vector<int>> paths;
#pragma omp parallel for schedule(dynamic) private(paths) shared(hsn, predecessorVector)
	for (int s = 0; s < size(); s++) {
		if (!borderNodes[s]) continue;
		int startIndex = size() * omp_get_thread_num();
		for (int i = startIndex; i < startIndex + size(); i++)
			predecessorVector[i] = std::vector<int>();
		localBFSAllPaths(s, clusters, predecessorVector + startIndex);
		for (int t = s + 1; t < size(); t++) {
			if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
			//paths.push_back({ &(vertices[t]) });
			//if (s == 19 && t == 28)
			//	std::cout << "This is the bad pair";
			paths = {};
			collectPaths(s, t, paths, predecessorVector + startIndex);
			for (int i = 0; i < paths.size(); i++)
				for (int j = 0; j < paths[i].size(); j++)
					hsn[paths[i][j]] = true;
			hsn[t] = true;
		}
		hsn[s] = true;
	}
	/*
	for (int s = 0; s < size(); s++) {
		if (!borderNodes[s]) continue;
		for (int t = s + 1; t < size(); t++) {
			if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
			std::vector<std::vector<int>> paths;
			//paths.push_back({ &(vertices[t]) });
			//if (s == 19 && t == 28)
			//	std::cout << "This is the bad pair";
			collectPaths(s, t, paths, predecessorVectors[s]);
			for (int i = 0; i < paths.size(); i++)
				for (int j = 0; j < paths[i].size(); j++)
					hsn[index(paths[i][j])] = true;
			hsn[t] = true;
		}
		hsn[s] = true;
	}*/
	delete[] predecessorVector;
	//return hsn;
}
 void Graph::findExternalNodes(std::vector<bool>& hsn, std::vector<int>& clusters, std::vector<bool>& borderNodes,
	std::vector<std::vector<bool>>& updatedClusters, int nClusters, std::vector<bool>* exn) {
	//std::vector<std::vector<bool>> exn(nClusters);
	for (int i = 0; i < nClusters; i++)
		exn[i].resize(size());
	std::vector<int>* predecessorVector = new std::vector<int>[size() * nThreads];
	std::vector<std::vector<int>> paths;
	std::vector<int> hsnInt(size());
	for (int i = 0; i < size(); i++)
		hsnInt[i] = int(hsn[i]);
#pragma omp parallel for schedule(dynamic) private(paths) shared(exn, updatedClusters, predecessorVector)
	for (int s = 0; s < size(); s++) {
		updatedClusters[clusters[s]][s] = true;
		if (!borderNodes[s]) continue;
		int startIndex = size() * omp_get_thread_num();
		for (int i = startIndex; i < startIndex + size(); i++)
			predecessorVector[i] = std::vector<int>();
		localBFSAllPaths(s, hsnInt, predecessorVector + startIndex);
		if (!borderNodes[s]) continue;
		for (int t = s + 1; t < size(); t++) {
			if (!borderNodes[t] || clusters[s] != clusters[t]) continue;
			//paths.push_back({ &(vertices[t]) });
			paths = {};
			collectPaths(s, t, paths, predecessorVector + startIndex);

			for (int i = 0; i < paths.size(); i++)
				for (int j = 1; j < paths[i].size(); j++) {
					if (clusters[paths[i][j]] != clusters[s]) {
						exn[clusters[s]][paths[i][j]] = true;
						updatedClusters[clusters[s]][paths[i][j]] = true;
					}
				}
		}
	}
	delete[] predecessorVector;
}
void Graph::localDeltas(std::vector<int>& clusters, std::vector<bool>& borderNodes, std::vector<std::vector<bool>>& updatedClusters, std::vector<bool>* externalNodes,
	double* localDeltas, std::vector<std::vector<int>>& clusterVector, std::vector<std::vector<Graph::equivalenceClass> >& classes, int nClusters,
	int maxCluster) {
	double** normalisedSigmas = new double*[maxCluster * nThreads];
	for (int i = 0; i < maxCluster * nThreads; i++) {
		normalisedSigmas[i] = new double[size()];
		for (int j = 0; j < size(); j++)
			normalisedSigmas[i][j] = 0;
	}
	int** distances = new int* [maxCluster * nThreads];
	for (int i = 0; i < maxCluster * nThreads; i++) {
		distances[i] = new int[size()];
		for (int j = 0; j < size(); j++)
			distances[i][j] = 0;
	}
	int** localSigmas = new int* [maxCluster * nThreads];
	for (int i = 0; i < maxCluster * nThreads; i++) {
		localSigmas[i] = new int[size()];
		for (int j = 0; j < size(); j++)
			localSigmas[i][j] = 0;
	}
	double* deltas = new double[size() * nThreads];
	int* sigmas = new int[size() * nThreads];
	std::vector<int>* predecessors = new std::vector<int>[size() * nThreads];
	for (int i = 0; i < size() * nThreads; i++) {
		deltas[i] = 0;
		sigmas[i] = 0;
	}
	std::vector<int> updatedClustersInt(size());
	classes.resize(nClusters);
	int* vertexDistances = new int[size() * nThreads];
	int* vertexLocalSigmas = new int[size() * nThreads];
#pragma omp parallel for schedule(dynamic) reduction(+: localDeltas[0:size()]) firstprivate \
(updatedClustersInt) shared(deltas, sigmas, distances, classes, predecessors, localSigmas, normalisedSigmas)
	for (int c = 0; c < nClusters; c++) {
		int startIndex = size() * omp_get_thread_num();
		int localityIndex = maxCluster * omp_get_thread_num();
		for (int i = localityIndex; i < localityIndex + maxCluster; i++) {
			for (int j = 0; j < size(); j++) {
				normalisedSigmas[i][j] = 0;
				distances[i][j] = 0;
				localSigmas[i][j] = 0;
			}
		}
		// if (c == 0) std::cout << omp_get_num_threads() << " threads\n";
		//normalisedSigmas = {};
		//normalisedSigmas.resize(clusterVector[c].size());
		//for (int i = 0; i < clusterVector[c].size(); i++)
			//normalisedSigmas[i].resize(size());
		for (int i = 0; i < size(); i++) {
			updatedClustersInt[i] = int(updatedClusters[c][i]);
		}
		for (int i = 0; i < clusterVector[c].size(); i++) {
			for (int j = startIndex; j < startIndex + size(); j++) {
				vertexDistances[j] = 0;
				vertexLocalSigmas[j] = 0;
			}
			int s = clusterVector[c][i];
			for (int j = startIndex; j < startIndex + size(); j++) {
				deltas[j] = 0;
				sigmas[j] = 0;
				predecessors[j].clear();
			}
			deltas[s + startIndex] = 1.0;
			sigmas[s + startIndex] = 1;
			//localDeltas[s] = 1.0;
			//normalisedSigmas[s] = 1;
			//vertexLocalSigmas[s] = 1;
		//	std::cout << sigmas[s];
			modifiedBrandesV1BFS(s, updatedClustersInt, predecessors + startIndex, deltas + startIndex, sigmas + startIndex,\
				localDeltas, vertexLocalSigmas + startIndex, vertexDistances + startIndex, externalNodes[clusters[s]]); // !
			for (int j = 0; j < size(); j++) {
				distances[localityIndex + i][j] = vertexDistances[startIndex + j];
				localSigmas[localityIndex + i][j] = vertexLocalSigmas[startIndex + j];
			}													
			
		}
		for (int i = 0; i < clusterVector[c].size(); i++) {
			int s = clusterVector[c][i];
			int minDistance = _I32_MAX;
			int minSigma = _I32_MAX;
			for (int j = 0; j < clusterVector[c].size(); j++) {
				int v = clusterVector[c][j];
				if (!borderNodes[v]) continue;
				if (distances[localityIndex + j][s] < minDistance) minDistance = distances[localityIndex + j][s];
				if (localSigmas[localityIndex + j][s] < minSigma) minSigma = localSigmas[localityIndex  + j][s];
			}
			for (int j = 0; j < clusterVector[c].size(); j++) {
				int v = clusterVector[c][j];
				if (!borderNodes[v]) continue;
				distances[localityIndex + j][s] -= minDistance;
				normalisedSigmas[localityIndex + j][s] = double(localSigmas[localityIndex + j][s]) / minSigma;
			}
		}
		classes[c] = findClasses(distances + localityIndex, normalisedSigmas + localityIndex, clusterVector[c], borderNodes);
	}

	for (int v = 0; v < size(); v++)
		localDeltas[v] /= 2;
	/*
	for (int s = 0; s < size(); s++) {
		int minDistance = _I32_MAX;
		int minSigma = _I32_MAX;
		for (int v = 0; v < size(); v++) {
			if (clusters[v] != clusters[s] || !borderNodes[v]) continue;
			if (distances[v][s] < minDistance) minDistance = distances[v][s];
			if (localSigmas[v][s] < minSigma) minSigma = localSigmas[v][s];
		}
		for (int v = 0; v < size(); v++) {
			if (clusters[v] != clusters[s] || !borderNodes[v]) continue;
			normalisedSigmas[v].resize(size(), 0);
			//distances[v].resize(size(), 0);
			distances[v][s] -= minDistance;
			normalisedSigmas[v][s] = double(localSigmas[v][s]) / minSigma;
		}
		localDeltas[s] /= 2;
	}
	*/
	for (int i = 0; i < maxCluster * nThreads; i++) {
		delete[] normalisedSigmas[i];
		delete[] distances[i];
		delete[] localSigmas[i];
	}
	delete[] normalisedSigmas;
	delete[] distances;
	delete[] localSigmas;
	delete[] deltas;
	delete[] sigmas;
	delete[] predecessors; 
	delete[] vertexDistances;
	delete[] vertexLocalSigmas;
}

Graph::equivalenceClass::equivalenceClass(int vertex, std::vector<int> _distances, std::vector<double> _sigmas) {
		indices.push_back(vertex);
		distances = _distances;
		sigmas = _sigmas;
	}
void Graph::equivalenceClass::addVertex(int vertex) {
		indices.push_back(vertex);
	}
int Graph::equivalenceClass::random() {
		return indices[rand() % indices.size()];
	}
std::vector<typename Graph::equivalenceClass> Graph::findClasses(int** distances, double** sigmas,
	std::vector<int>& vectorCluster, std::vector<bool>& borderNodes) {
	//std::vector<std::vector<equivalenceClass>> classes;
	std::vector<equivalenceClass> clusterClasses;
	for (int i = 0; i < vectorCluster.size(); i++) {
		int s = vectorCluster[i];
		bool foundClass = false;
		std::vector<int> BNdists;
		std::vector<double> BNsigmas;
		for (int j = 0; j < vectorCluster.size(); j++) {
			int v = vectorCluster[j];
			if (!borderNodes[v]) continue;
			BNdists.push_back(distances[j][s]);
			BNsigmas.push_back(sigmas[j][s]);
		}
		for (int j = 0; j < clusterClasses.size(); j++) {
			//BNdists = {};
			//BNsigmas = {};
			bool suitableClass = true;
			int nNode = 0;
			for (int k = 0; k < vectorCluster.size(); k++) {
				int v = vectorCluster[k];
				if (!borderNodes[v]) continue;
				if (distances[k][s] != clusterClasses[j].distances[nNode] || sigmas[k][s] != clusterClasses[j].sigmas[nNode]) suitableClass = false;
				nNode++;
			}
			if (suitableClass) {
				clusterClasses[j].addVertex(s);
				foundClass = true;
			}
		}
		if (!foundClass)
			clusterClasses.push_back(equivalenceClass(s, BNdists, BNsigmas));
	}
	return clusterClasses;
}
void Graph::globalDeltas(std::vector<std::vector<equivalenceClass>>& classes, std::vector<int>& clusters, int nClusters, double* res) {
	//std::vector<double> res(size());
	std::vector<int>* predecessors = new std::vector<int>[size() * nThreads];
	double* deltaContribs = new double[size() * nThreads];
#pragma omp parallel for schedule(dynamic) reduction(+: res[0:size()]) firstprivate(deltaContribs) shared(predecessors)
	for (int c = 0; c < nClusters; c++) {
		for (int i = 0; i < classes[c].size(); i++) {
			if (c == 0 && i == 0) std::cout << omp_get_num_threads() << " threads\n";
			int pivot = classes[c][i].indices[0];
			int startIndex = size() * omp_get_thread_num();
			for (int j = startIndex; j < startIndex + size(); j++) {
				predecessors[j] = std::vector<int>();
				deltaContribs[j] = 0;
			}
			modifiedBrandesV2BFS(pivot, clusters, predecessors + startIndex, deltaContribs + startIndex, nClusters);
			for (int s = 0; s < size(); s++) {
				res[s] += deltaContribs[s + startIndex] * classes[c][i].indices.size() / 2;
			}
		}
	}
	delete[] predecessors;
	delete[] deltaContribs;
	//return;
	//return res;
}



Graph::Graph(int size) {
	nThreads = env.nThreads;
	row_index.resize(size + 1);
}
//Graph(std::vector<Vertex> v, int** ws) {
//	vertices = v;
//	weights = ws;
//}
Graph::Graph(const Graph& g) {
	nThreads = g.nThreads;
	adjacencies = g.adjacencies;
	weights = g.weights;
	row_index = g.row_index;
}
Graph Graph::operator=(const Graph& g) {
	nThreads = g.nThreads;
	adjacencies = g.adjacencies;
	weights = g.weights;
	row_index = g.row_index;
	return *this;
}
int Graph::weight(int v1, int v2) {
	for (int i = row_index[v1]; i < row_index[v1 + 1]; i++) {
		if (adjacencies[i] == v2) return weights[i];
	}
	return 0;
}



void Graph::BFSAllPaths(int startingVertex, std::vector<std::vector<std::vector<int>>>& _paths) {
	//std::vector<bool> markedVertices(size());
	std::vector<int>* predecessors = new std::vector<int>[size()];
	std::vector<std::vector<std::vector<int>>> paths(size());
	//markedVertices[startingVertex] = true;
	std::vector<int> path = {};
	path.push_back(startingVertex);
	std::vector<int> distances(size());
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) distances[i] = 0;
		else distances[i] = UNREACHABLE;
	}
	std::deque<int> q;
	q.push_back(startingVertex);
	while (!q.empty()) { //
		int u = q.front();
		q.pop_front();
		for (int i = row_index[u]; i < row_index[u + 1]; i++) {
			int it = adjacencies[i];
			if (distances[it] == UNREACHABLE) {
				//markedVertices[it] = true;
				//path.push_back(it);
				distances[it] = distances[u] + 1;
				//predecessors[it] = u;
				q.push_back(it);
			}
			if (distances[u] + 1 == distances[it]) {
				predecessors[it].push_back(u);
			}
		}
	} //
	paths[startingVertex].push_back({});
	for (int i = 0; i < size(); i++) {
		if (i == startingVertex) continue;
		paths[i].push_back({ i });
		std::vector<int> currPaths = { 0 };
		collectPaths(startingVertex, i, paths[i], predecessors);
	}
	_paths = paths;
	delete[] predecessors;
}


int Graph::sumAllWeights(std::vector<int> clusters, int cluster, int vertex, int* incC, int* incV, int* between) {
	int res = 0;
	if (incC != nullptr) *incC = 0;
	if (incV != nullptr) *incV = 0;
	if (between != nullptr) *between = 0;
	if (cluster == -1) {
		clusters = {};
		clusters.resize(size());
		cluster = 0;
	}
	for (int i = 0; i < size(); i++) {
		if (i == vertex) {
			for (int j = row_index[i]; j < row_index[i + 1]; j++) {
				int it = adjacencies[j];
				if (clusters[it] == cluster) {
					if (between != nullptr) *between += weights[j];
				}
				if (incV != nullptr) *incV += weights[j];
			}

		}
		if (clusters[i] == cluster) {
			for (int j = row_index[i]; j < row_index[i + 1]; j++) {
				int it = adjacencies[j];
				if (clusters[it] == cluster && it <= i) res += weights[j];
				else if (incC != nullptr) *incC += weights[j];
			}
		}
	}
	return res;
}
std::vector<int> Graph::louvain(int& nClusters, int desiredNClusters) {
	std::vector<int> clusters(size());
	for (int i = 0; i < size(); i++)
		clusters[i] = i;
	std::vector<int> communities = clusters;
	nClusters = size();
	int currNClusters = size();

	Graph currNetwork(*this);
	//currNetwork.vertices = vxs;
	double m = sumAllWeights(clusters);
	while (true) {
		//First stage
		bool changed = false;
		std::vector<int> oldClusters = clusters;
		std::vector<bool> communityRemoved(size());
		for (int i = 0; i < currNetwork.size(); i++) {
			double maxModularityGain = 0; int argmmg = -1;
			std::vector<bool> clusterProcessed(size());
			bool singleton = true;
			double modularityGain;
#pragma omp parallel for schedule(dynamic) shared(clusterProcessed, modularityGain, argmmg)
			for (int j = currNetwork.row_index[i]; j < currNetwork.row_index[i + 1]; j++) {
				int it = currNetwork.adjacencies[j];
				//	std::cout << currNetwork.index(it->data) << std::endl;
				if (clusterProcessed[communities[it]]) continue;
				if (communities[it] == communities[i]) {
					if (it != i) singleton = false;
					continue;
				}
				int sin, stot, ki, kiin;
				sin = currNetwork.sumAllWeights(communities, communities[it], i, &stot, &ki, &kiin);
				modularityGain = (double(sin + kiin) / (2 * m) - (double(stot + ki) / (2 * m)) * (double(stot + ki) / (2 * m))) -
					(double(sin) / (2 * m) - (double(stot) / (2 * m)) * (double(stot) / (2 * m)) - (double(ki) / (2 * m)) * (double(ki) / (2 * m)));
				if (modularityGain > maxModularityGain) {
					maxModularityGain = modularityGain;
					argmmg = communities[it];
				}
				clusterProcessed[communities[it]] = true;
			}
			if (argmmg != -1) {
				if (singleton) {
					currNClusters--;
					communityRemoved[i] = true; //
				}
				for (int j = 0; j < size(); j++)
					if (oldClusters[j] == communities[i]) clusters[j] = argmmg;
				communities[i] = argmmg;
				changed = true;
			}
		}
		//Second stage
		if (!changed) return clusters;
		nClusters = currNClusters;
		////int** newws = new int* [nClusters];
		//for (int i = 0; i < nClusters; i++) {
		//	newws[i] = new int[nClusters];
		//}
		//for (int i = 0; i < nClusters; i++)
		//	newvxs[i].data.first = i;
		Graph newNetwork(nClusters);

		std::vector<int> newCommunities(size());
		std::vector<bool> communityHadBeenRemoved = communityRemoved;
		std::vector<bool> oldCommunityProcessed(size());
		std::vector<int> transformTable(size());
		std::fill(transformTable.begin(), transformTable.end(), -1);
		int currCommunity = 0;

		for (int i = 0; i < currNetwork.size(); i++) {
			currCommunity = communities[i];
			int sin, kiin;
			if (communities[i] >= nClusters)
				if (transformTable[communities[i]] != -1) currCommunity = transformTable[communities[i]];
			if (!oldCommunityProcessed[communities[i]]) {
				if (communities[i] >= nClusters) {
					if (transformTable[communities[i]] != -1) currCommunity = transformTable[communities[i]];
					else {
						bool foundSlot = false;
						for (int j = 0; j < nClusters; j++)
							if (communityRemoved[j]) {
								currCommunity = j;
								communityRemoved[j] = false;
								transformTable[communities[i]] = j;
								foundSlot = true;
								break;
							}
						if (!foundSlot)
							std::cout << "Slot not found";
					}
				}
				sin = sumAllWeights(communities, currCommunity);
				newNetwork.addEdge(currCommunity, currCommunity, sin);
			}
			for (int k = currNetwork.row_index[i]; k < currNetwork.row_index[i + 1]; k++) {
				int it = currNetwork.adjacencies[k];
				if (communities[it] == communities[i]) continue;
				int otherCommunity = communities[it];
				if (communities[it] >= nClusters) {
					if (transformTable[communities[it]] != -1) otherCommunity = transformTable[communities[it]];
					else {
						bool foundSlot = false;
						for (int j = 0; j < nClusters; j++)
							if (communityRemoved[j]) {
								otherCommunity = j;
								communityRemoved[j] = false;
								transformTable[communities[it]] = j;
								foundSlot = true;
								break;
							}
						if (!foundSlot)
							std::cout << "Slot not found";
					}
				}
				if (!newNetwork.areNeighbours(currCommunity, otherCommunity)) {
					newNetwork.addEdge(currCommunity, otherCommunity, 0);
					//newNetwork.weights[currCommunity][otherCommunity] = 0;
				}
				newNetwork.assignWeight(currCommunity, otherCommunity, newNetwork.weight(currCommunity, otherCommunity) + currNetwork.weight(i, it)); // !
				//foundLink[otherCommunity] = true;
				//if(!currCommuityProcessed && ) newNetwork.addEdge(currCommunity, , 1)
				//currNetwork.weights[currCommunity][
			}
			//if (!oldCommunityProcessed[communities[i]]) {
			//	newNetwork.weights[currCommunity][currCommunity] = sin;
			//}
			oldCommunityProcessed[communities[i]] = true;
		}

		for (int i = 0; i < nClusters; i++)
			communities[i] = i;
		for (int i = 0; i < size(); i++)
			if (clusters[i] >= nClusters) clusters[i] = transformTable[clusters[i]];

		currNetwork = newNetwork;
		if (nClusters < desiredNClusters) return clusters;
		/*
		for (int i = 0; i < newNetwork.size(); i++) {
			for (typename List<int>::iterator it = newNetwork.vertices[i].adjacencyList.begin(); it != newNetwork.vertices[i].adjacencyList.end(); ++it) {
				currNetwork.vertices[i].adjacencyList.push_front(&(currNetwork.vertices[newNetwork.it]));
			}
		}

		for (int i = 0; i < nClusters; i++) {
			for (int j = 0; j < nClusters; j++) {
				currNetwork.weights[i][j] = newNetwork.weights[i][j];
			}
			delete[] newNetwork.weights[i];
		}
		delete[] newNetwork.weights;
		*/
	}


	return clusters;
}
//double modularity(std::vector<int> communities) {
//	double m = sumAllWeights(clusters);

//}
void Graph::adoptSingletons(std::vector<int>& clusters, int& nClusters) {
	for (int i = 0; i < size(); i++) {
		bool singleton = true;
		for (int j = row_index[i]; j < row_index[i + 1]; j++)
			if (clusters[i] == clusters[adjacencies[j]])
				singleton = false;
		if (singleton)
			clusters[i] = clusters[adjacencies[row_index[i]]];
	}
	std::vector<bool> slotFilled;
	for (int i = 0; i < nClusters; i++) {
		if (std::find(clusters.begin(), clusters.end(), i) != clusters.end()) slotFilled.push_back(true);
		else slotFilled.push_back(false);
	}
	for (int i = nClusters - 1; i >= 0; i--) {
		if (slotFilled[i]) {
			bool slotFound = false;
			for (int j = 0; j < nClusters; j++)
				if (!slotFilled[j]) {
					for (int k = 0; k < size(); k++)
						if (clusters[k] == i) clusters[k] = j;
					slotFilled[j] = true;
					slotFound = true;
					break;
				}
			if (!slotFound) break;
		}
		nClusters--;
	}
}
std::vector<double> Graph::brandes(std::vector<int> clusters) {
	if (clusters.size() == 0) {
		for (int i = 0; i < size(); i++)
			clusters.push_back(0);
	}
	std::vector<double> res(size());
	std::vector<std::vector<int>> predecessors(size());
	//std::vector<std::vector<int>> S(size(), vector<int>(size(), 0));

	for (int i = 0; i < size(); i++) {
		std::vector<double> deltas(size());
		std::vector<int> sigmas(size());
		std::fill(predecessors.begin(), predecessors.end(), std::vector<int>());
		deltas[i] = 1.0;
		sigmas[i] = 1;
		brandesBFS(i, clusters, predecessors, deltas, sigmas, res);
	}
	for (int i = 0; i < size(); i++)
		res[i] = res[i] / 2;
	return res;
}
std::vector<double> Graph::brandesNaive() {
	std::vector<double> res(size());
	//std::vector<std::vector<int>> S(size(), vector<int>(size(), 0));
	std::vector<std::vector<std::vector<std::vector<int>>>> paths(size());
	for (int i = 0; i < size(); i++) {
		BFSAllPaths(i, paths[i]);
	}
	for (int v = 0; v < size(); v++) {
		for (int s = 0; s < size(); s++) {
			if (s == v) continue;
			for (int t = s + 1; t < size(); t++) {
				int sigmaV = 0;
				if (t == v) continue;
				for (int i = 0; i < paths[s][t].size(); i++)
					for (int j = 0; j < paths[s][t][i].size(); j++)
						if (paths[s][t][i][j] == v) sigmaV++;
				res[v] += double(sigmaV) / paths[s][t].size();
			}
		}
	}
	return res;
}
std::vector<double> Graph::fastBC(std::vector<double>& stageTimes) { // clustering 0, border nodes 1, hsn 2, external nodes 3, \
	local deltas + classes 4, global deltas 5
	double begin_time = omp_get_wtime();
	std::vector<double> res(size());
	int nClusters;
	//int dnc = pow(size(), 0.5);
	std::vector<int> clusters = louvain(nClusters);
	adoptSingletons(clusters, nClusters);
	
	std::vector<std::vector<int>> clusterVector(nClusters);
	double checkpoint = omp_get_wtime();
	stageTimes[0] = checkpoint - begin_time;
	for (int i = 0; i < size(); i++) {
		clusterVector[clusters[i]].push_back(i);
	}
	int maxCluster = 0;
	for (int i = 0; i < nClusters; i++) {
		int verticesInCluster = clusterVector[i].size();
		if (verticesInCluster > maxCluster) maxCluster = verticesInCluster;
		// std::cout << i << ": " << verticesInCluster << std::endl;
	}

	std::vector<bool> borderNodes(size());
	findBorderNodes(clusters, borderNodes);
	stageTimes[1] = omp_get_wtime() - checkpoint;
	checkpoint = omp_get_wtime();
	std::vector<bool> hsn(size());
	buildHSN(clusters, borderNodes, hsn);
	stageTimes[2] = omp_get_wtime() - checkpoint;
	checkpoint = omp_get_wtime();
	std::vector<std::vector<bool>> updatedClusters(nClusters);
	for (int i = 0; i < nClusters; i++)
		updatedClusters[i].resize(size());
	std::vector<bool>* externalNodes = new std::vector<bool>[nClusters];
	findExternalNodes(hsn, clusters, borderNodes, updatedClusters, nClusters, externalNodes);
	stageTimes[3] = omp_get_wtime() - checkpoint;
	checkpoint = omp_get_wtime();
	double* locDeltas = new double[size()];
	for (int i = 0; i < size(); i++)
		locDeltas[i] = 0;
	//std::vector<std::vector<int>> distances(size());
	//std::vector<std::vector<double>> normalisedSigmas(size());
	std::vector<std::vector<Graph::equivalenceClass>> classes;
	localDeltas(clusters, borderNodes, updatedClusters, externalNodes, locDeltas, clusterVector, classes, nClusters, maxCluster);
	stageTimes[4] = omp_get_wtime() - checkpoint;
	checkpoint = omp_get_wtime();
	//std::vector<std::vector<equivalenceClass>> classes = findClasses(distances, normalisedSigmas, clusters, borderNodes, nClusters);
	double* globDeltas = new double[size()];
	for (int i = 0; i < size(); i++)
		globDeltas[i] = 0;
	globalDeltas(classes, clusters, nClusters, globDeltas);
	for (int v = 0; v < size(); v++) {
		res[v] += locDeltas[v];
		res[v] += globDeltas[v];
	}
	delete[] externalNodes;
	delete[] locDeltas;
	delete[] globDeltas;
	stageTimes[5] = omp_get_wtime() - checkpoint;
	return res;
}
void Graph::addEdge(int v1, int v2, int weight) {
	adjacencies.insert(adjacencies.begin() + row_index[v1 + 1], v2);
	weights.insert(weights.begin() + row_index[v1 + 1], weight);
	for (int i = v1 + 1; i < row_index.size(); i++)
		row_index[i]++;
	if (v1 == v2) return;
	adjacencies.insert(adjacencies.begin() + row_index[v2 + 1], v1);
	weights.insert(weights.begin() + row_index[v2 + 1], weight);
	for (int i = v2 + 1; i < row_index.size(); i++)
		row_index[i]++;
}
void Graph::assignWeight(int v1, int v2, int weight) {
	for (int i = row_index[v1]; i < row_index[v1 + 1]; i++)
		if (adjacencies[i] == v2) {
			weights[i] = weight;
			break;
		}
	if (v1 == v2) return;
	for (int i = row_index[v2]; i < row_index[v2 + 1]; i++)
		if (adjacencies[i] == v1) {
			weights[i] = weight;
			break;
		}
}
bool Graph::areNeighbours(int v1, int v2) {
	for (int i = row_index[v1]; i < row_index[v1 + 1]; i++)
		if (adjacencies[i] == v2)
			return true;
	return false;
}
void Graph::removeEdge(int v1, int v2) {
	for (int i = row_index[v1]; i < row_index[v1 + 1]; i++)
		if (adjacencies[i] == v2) {
			adjacencies.erase(adjacencies.begin() + row_index[i]);
			weights.erase(weights.begin() + row_index[i]);
			for (int i = v1 + 1; i < row_index.size(); i++)
				row_index[i]--;
		}
	if (v1 == v2) return;
	for (int i = row_index[v2]; i < row_index[v2 + 1]; i++)
		if (adjacencies[i] == v1) {
			adjacencies.erase(adjacencies.begin() + row_index[i]);
			weights.erase(weights.begin() + row_index[i]);
			for (int i = v2 + 1; i < row_index.size(); i++)
				row_index[i]--;
		}
}