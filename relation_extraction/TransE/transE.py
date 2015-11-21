#!/usr/bin/python env
#-*- coding:utf-8 -*-

import sys
import numpy as np
import scipy as sp
import collections
import math
import random

normal = lambda x,miu,sigma:1.0/math.sqrt(2*math.pi)/sigma*math.exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma))
vec_len = lambda vec:math.sqrt(sum(map(lambda x:x**2,vec)))

def randn(miu, sigma, minV, maxV):
	while True:
		x = random.uniform(minV, maxV)
		y = normal(x, miu, sigma)
		dscope = random.uniform(0.0, normal(miu, miu, sigma))
		if dscope <= y:
			break
	return x


class TransE:
	def __init__(self, dim=100, rate=0.001, margin=1):
		self.l1_flag = True 
		self.entity2id={}
		self.id2entity={}
		self.relation2id={}
		self.id2relation={}
		self.entityNum = 0
		self.relationNum = 0
		self.dim = dim 
		self.rate = rate
		self.margin = margin
		self.loss = 0.0 #loss function value
		self.method = 0
		self.left_entity = collections.defaultdict(dict)
		self.right_entity = collections.defaultdict(dict)
		self.left_num = {}
		self.right_num = {}
		#self.train=[]
		self.train = collections.defaultdict(dict)
		self.test = []
		self.fb_h = []
		self.fb_t = []
		self.fb_l = []
		#self.entity_vec, vec_tmp
		#self.relation_vec, vec_tmp
	def process_line_entity2id(self, lines):
		assert len(lines)==2
		entity = lines[0]
		eid = int(lines[1])
		self.entity2id[entity] = eid
		self.id2entity[eid] = entity
		self.entityNum += 1
	def process_line_relation2id(self, lines):
		assert len(lines)==2
		relation = lines[0]
		lid = int(lines[1])
		self.relation2id[relation] = lid
		self.id2relation[lid] = relation
		self.relationNum += 1
	def process_line_train(self, lines):
		assert len(lines)==3
		e1 = lines[0]
		e1id = self.entity2id[e1]
		e2 = lines[1]
		e2id = self.entity2id[e2]
		rel = lines[2]
		if not self.relation2id.has_key(rel):
			self.relation2id[rel] = self.relationNum
			self.id2relation[self.relationNum] = rel
			self.relationNum += 1
		lid = self.relation2id[rel]
		self.left_entity[lid][e1id] = self.left_entity[lid].get(e1id, 0) + 1
		self.right_entity[lid][e2id] = self.right_entity[lid].get(e2id, 0) + 1
		#self.train.append((e1id, lid, e2id))
		self.train[(e1id, lid)][e2id]=1
		self.fb_h.append(e1id)
		self.fb_t.append(e2id)
		self.fb_l.append(lid)
	def process_line_test(self, lines):
		pass
	def read_file(self, dataset, filename):
		"""called by prepare_data"""
		prefix = "../data/"
		#if dataset=="WN18":
		#		elif dataset=="FB15k":
		filepath = prefix + dataset + "/" + filename
		with open(filepath) as fin:
			lineN = 0
			for line in fin:
				lineN += 1
				if lineN%10000==0:
					print lineN,"\t", line
				lines = line.strip().split()
				if filename=="entity2id.txt":
					self.process_line_entity2id(lines)
				elif filename=="relation2id.txt":
					self.process_line_relation2id(lines)
				elif filename=="train.txt":
					self.process_line_train(lines)
				elif filename=="test.txt":
					self.process_line_test(lines)
				else:
					raise IOError, 'cannot process line %d, content: %s' % (lineN, line)
			print "%d lines, %d entities, %d relations." % (lineN, self.entityNum, self.relationNum)
			if filename=="train.txt":
				print len(self.left_entity.keys())
				f1 = lambda x:sum(self.left_entity[x].values())*1.0/len(self.left_entity[x].keys())
				f2 = lambda x:sum(self.right_entity[x].values())*1.0/len(self.right_entity[x].keys())
				l1 = map(f1, self.left_entity.keys())
				print l1
				l2 = map(f2, self.right_entity.keys())
				print l2
				print len(l1), len(l2)
				print len(self.id2relation.keys())
				for i,k in enumerate(self.id2relation.keys()):
					self.left_num[k] = l1[i]
					self.right_num[k] = l2[i]
#print self.left_num[:50]
#				print self.right_num[:50]
	def prepare_data(self, dataset):
		self.read_file(dataset, "entity2id.txt")
		self.read_file(dataset, "relation2id.txt")
		self.read_file(dataset, "train.txt")
		self.entity_vec = np.zeros((self.entityNum, self.dim), dtype=np.float64)
		self.relation_vec = np.zeros((self.relationNum, self.dim), dtype=np.float64)
		self.entity_tmp = np.zeros((self.entityNum, self.dim), dtype=np.float64)
		self.relation_tmp = np.zeros((self.relationNum, self.dim), dtype=np.float64)
		print "Prepare data done..."
		print np.shape(self.entity_vec)
		print np.shape(self.relation_vec)
	def calc_sum(self, e1id, e2id, lid):
		delta_vec = self.entity_vec[e2id] - self.entity_vec[e1id] - self.relation_vec[lid] 
		if self.l1_flag:
			return reduce(lambda x,y:math.fabs(x)+math.fabs(y), delta_vec)
		else:
			return sum(delta_vec**2)
	def gradient(self, e1aid,  e2aid,  raid, e1bid, e2bid, rbid):
		"""(e1a, ra, e2a) -> (e1b, rb, e2b)"""
		for i in range(self.dim):
			x = 2*(self.entity_vec[e2aid][i] - self.entity_vec[e1aid][i] - self.relation_vec[raid][i])
			if self.l1_flag:
				if x > 0:
					x = 1
				else:
					x = -1
			self.relation_tmp[raid][i] -= -1* self.rate * x
			self.entity_tmp[e1aid][i] -= -1* self.rate * x
			self.entity_tmp[e2aid][i] += -1* self.rate * x
			x = 2*(self.entity_vec[e2bid][i] - self.entity_vec[e1bid][i] - self.relation_vec[rbid][i])
			if self.l1_flag:
				if x > 0:
					x = 1
				else:
					x = -1
			self.relation_tmp[rbid][i] -= -1* self.rate * x
			self.entity_tmp[e1bid][i] -= -1* self.rate * x
			self.entity_tmp[e2bid][i] += -1* self.rate * x
	def train_kb(self, e1aid, e2aid, raid, e1bid, e2bid, rbid):
		suma = self.calc_sum(e1aid, e2aid, raid)
		sumb = self.calc_sum(e1bid, e2bid, rbid)
		if suma + self.margin > sumb:
			self.loss += margin + suma - sumb
			self.gradient(e1aid, e2aid, raid, e2aid, e2bid, rbid)
	def norm(self, vec):
		x = vec_len(vec)
		if x>1:
			#vec = map(lambda y:y/x, vec)
			for i in range(len(vec)):
				vec[i] = vec[i]/x
	def bfgs(self):
		nbatches = 100
		nepoch = 100
		batchsize = len(self.fb_h)/nbatches
		for epoch in range(nepoch):
			self.loss = 0
			for batch in range(nbatches):
				self.relation_tmp = self.relation_vec
				self.entity_tmp = self.entity_vec
				for k in range(batchsize):
					i = random.randint(0,len(self.fb_h)-1) #???
					j = random.randint(0, self.entityNum-1)
#print "i=%d,j=%d,len(fb_l)=%d" % (i,j,len(self.fb_l))
					pr = 1000 * self.right_num[self.fb_l[i]] /( self.right_num[self.fb_l[i]] + self.left_num[self.fb_l[i]])
					if self.method == 0:
						pr = 500
					if random.randint(0,1000)%1000 < pr:
						while self.train[(self.fb_h[i],self.fb_l[i])].has_key(j):
							j = random.randint(0, self.entityNum)
						self.train_kb(self.fb_h[i], self.fb_t[i],self.fb_l[i], self.fb_h[i], j, self.fb_l[i] ) # just difference in right entity
					else:
						while self.train[(j,self.fb_l[i])].has_key(self.fb_t[i]):
							j = random.randint(0, self.entityNum)
						self.train_kb(self.fb_h[i], self.fb_t[i],self.fb_l[i], j, self.fb_t[i], self.fb_l[i] ) # just difference in left entity
					self.norm(self.relation_tmp[self.fb_l[i]])
					self.norm(self.entity_tmp[self.fb_h[i]])
					self.norm(self.entity_tmp[self.fb_t[i]])
					self.norm(self.entity_tmp[j])
				self.relation_vec = self.relation_tmp
				self.entity_vec = self.entity_tmp
			print "Epoch:%d, loss:%.4f" % (epoch, self.loss)

	def run(self, dim, rate, margin, method):
		v = randn(0,1.0/self.dim, -6.0/math.sqrt(self.dim), 6.0/math.sqrt(self.dim))
		self.relation_vec.fill(v)
		self.entity_vec.fill(v)
		for i in range(self.relationNum):
			self.norm(self.entity_vec[i])
		self.bfgs()

if __name__=="__main__":
	rate = 0.001
	margin = 1.0
	dim = 100
	method = "bern"
	# method = "unif"
	dataset = "WN18"
	m = TransE(dim, rate, margin)
	m.prepare_data(dataset)
	m.run(dim, rate, margin, method)
