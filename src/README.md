## Src

This file contains the code for this project, including the DSL, interpreter, models, training and environment,

### TODO LIST

- Implement the DSL (Note [] means interpret, v is the word (implicit))	

	- Concat (e1, e2, e3, ...) = Concat ( [e1] , [e2], [e3], ...)

	- ConstStr (c) = c

	- Substr (k1, k2) = v[p1...p2] where p is k if k > 0, else len(v) + p

	- GetSpan(r1, i1, y1, r2, i2, y2) = starting at i1th match of r1, y1 says start or end, end at i2th match of r2, y2 ""

	- _Nested Expressions_ [n1(n2)] = [n1]_v1_ where v1 = [n2]

	- GetToken(t, i) = ith match of t from beginning (end if i < 0)

	- GetUpto(r) = v[0...i] where i is the index at end of match of r

	- GetFrom(r) = v[j...len(v)] where j is end of last match of r 

	- GetFirst(t, i) = Concat first i matches of t

	- GetAll(t) = Concat all matches of t

	- ToCase(s) = upper or lower

	- Trim() = removes whitespace from around the string

	- Replace(delta1, delta2) = replaces regex delta1 with delta2

- Implement MDP formulation (environment)
	
	- Observations: random I/O generation, and random program sampling 

	- action: program from agent

	- reward function : binary 0 or 1 (matches or not)


- Implement the models
	
	- Base model from Robsutfill / Reinforce

	- AC model (LSTM and GRU approach)

	- SAC model

	- Training code for models ^^^

- Run basic expirements
	
	- Try to recreate REINFORCE results from Robustfill

	- AC approach

	- SAC approach

- Bells and Whistles
	
	- Implement replay buffer / other techniques for sparse reward handling

	- Command line / QOL changes

	- Python Notebook for running