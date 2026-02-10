import json

class Node:
    def __init__(self, val=None):
        self.val = val
        self.children = []
        self.map = {}
        self.end = False
        self.data = None

class Trie:
    def __init__(self, words, data_map=None):
        self.root = Node()
        self.data_map = data_map or {}
        self.build(words)
    
    def build(self, words):
        for word in words:
            cur = self.root
            for c in word:
                if c not in cur.map:
                    nxt = Node(val=c)
                    cur.map[c] = len(cur.children)
                    cur.children.append(nxt)
                cur = cur.children[cur.map[c]]
            cur.end = True
            cur.data = self.data_map.get(word)
    
    def contains(self, word):
        cur = self.root
        for c in word:
            if c not in cur.map:
                return False
            cur = cur.children[cur.map[c]]
        return cur.end
    
    def complete(self, word):
        cur = self.root
        for c in word:
            if c not in cur.map:
                return []
            cur = cur.children[cur.map[c]]
        
        results = []
        if cur.end:
            results.append(cur.data)
        results.extend(self.suffixes(cur))
        return results
    
    def suffixes(self, node):
        suf = []
        
        def dfs(curr):
            if curr.end:
                suf.append(curr.data)
            for child in curr.children:
                dfs(child)
        
        for child in node.children:
            dfs(child)
        
        return suf


if __name__ == "__main__":
    #load tickers
    with open('tickers.json', 'r') as f:
        ticker_data = json.load(f)
    
    #create a mapping
    tickers = []
    data_map = {}
    for key, value in ticker_data.items():
        ticker = value['ticker']
        tickers.append(ticker)
        data_map[ticker] = value
    
    t = Trie(tickers, data_map)
    
    print(t.complete('TS'))
