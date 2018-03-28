import pandas as pd
import numpy as np

from agent import Agent
from brain import Brain
from env import Environment
from log import Log
from pool import Pool

from consts import *
import utils, sys, json, random, torch

import argparse

from tqdm import tqdm

#==============================
def is_time(epoch, trigger):
	return (trigger > 0) and (epoch % trigger == 0)

np.set_printoptions(threshold=np.inf)

#==============================
parser = argparse.ArgumentParser()
parser.add_argument('-ff', type=float, help="feature factor")
args = parser.parse_args()

if args.ff:
	FEATURE_FACTOR = args.ff
	print("FEATURE_FACTOR = {}".format(FEATURE_FACTOR))

#==============================
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#==============================
data = torch.load(DATA_FILE)
label = torch.load(LABEL_FILE)
init = torch.load(INIT_FILE)

data_val = torch.load(DATA_VAL_FILE)
label_val = torch.load(LABEL_VAL_FILE)
init_val = torch.load(INIT_VAL_FILE)

print("Using", DATA_FILE, "with", len(data), "samples.")

#==============================
pool  = Pool(POOL_SIZE)
env   = Environment(data, label, init, FEATURE_FACTOR)
brain = Brain(pool)
agent = Agent(env, pool, brain)
log   = Log(data_val, label_val, init_val, FEATURE_FACTOR, brain)
#==============================
epoch_start = 0

if not BLANK_INIT:
	print("Loading progress..")
	brain._load()

	with open('run.state', 'r') as file:
		save_data = json.load(file)

	epoch_start = save_data['epoch']

brain.update_lr(epoch_start)
agent.update_epsilon(epoch_start)

#==============================
print("Initializing pool..")
for i in range(POOL_SIZE // AGENTS):
	utils.print_progress(i, POOL_SIZE // AGENTS)
	agent.step()

pool.cuda()
	
print("Starting..")
for epoch in range(epoch_start + 1, TRAINING_EPOCHS + 1):
	# SAVE
	if is_time(epoch, SAVE_EPOCHS):
		brain._save()

		save_data = {}
		save_data['epoch'] = epoch

		with open('run.state', 'w') as file:
			json.dump(save_data, file)

	# SET VALUES
	if is_time(epoch, EPSILON_UPDATE_EPOCHS):
		agent.update_epsilon(epoch)

	if is_time(epoch, LR_SC_EPOCHS):
		brain.update_lr(epoch)

	if is_time(epoch, LOG_PERF_EPOCHS):
		print ('[Test] epoch {}'.format(epoch))
		log.log_perf(epoch)

	# TRAIN
	brain.train()
	
	for i in range(EPOCH_STEPS):
		agent.step()
