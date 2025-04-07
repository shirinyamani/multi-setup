# Lessons:

## Lesson 1: Early stages of Optimizer
In early stages of optimozer, we are just learning very simple biases on use this toke or not use this token, basically telling us this toke is appreared or not, and so the gradient of these examples are very much like one another and highly correlated! and later in the optimization when the optimizer already learnt the simple stuff, thats where the actual works starts, when the gradients are somehow de-corrolated, that's when it can some how offer statistical power!

## Lesson2: Grad Accumulation 
if we wanna simulate a very large batch size of 0.5M, 
well first of all this 0.5M in the number of tokens, so for exp if we have 
```python
train_loader = DataLoader(B=16, T=1024) #we have 16 samples x 1024 tokens/sample = 16,384  tokens per forward backward
```
so for instance if you wanna simulate 0.5M ---> 2^19(in number of tokens)
```bash
2**19 / (16 * 1024) = 32 ---> grad_accum_step_size
```
so all in one:
```python
total_desired_batch_size = 524288 #2^19 ~ 0.5M in num tokens
B = 16 #per device batch size
T = 1024 #seq_len (number of tokens) per Batch
assert total_desired_batch_size // (B * T), "make sure your desired total_batch_size is divisible by the B * T"
grad_accum = total_desired_batch_size // (B * T) # 32 ---> meaning 32 fwd/bcward then a single optim update!
#therefore; 
tokens_proccessed = B * T * grad_accum_steps
```

## Lesson 3: DDP 
we have 8 gpus, so we gonna initiate 8 processes, and each of them will work on a slightly different part of the date, then once this is done, and they all calculated the gradients on their corresponding chunk of data, we gonna do `allreduce` to get the avrg of all the calculated gradients and that's how they gonna be collaborating on the computation here!

in this code that we will be running it using `torchrun` will make sure we run all the 8 processes in parallel and when it does in sets some env variables as follows; 

Definitions:
- `ddp_rank` : each process will run exact same code n roughly at the same time (note that which finish running the code depends on the way OS schedule the jobs so if you see for exp process numb 5 finished earlier that 0, dont be scared! cuz at the end `allreduce` will distribute the average of the grad in backward among all the gpus!) cuz we wanna make sure they all run on different part of the data not same!
- `ddp_local_rank` : used in multi-node setting-- the rank of the gpu on the each node
- `ddp_world_size`: total num of processes 

```python
from torch.dirstributed import init_process_group
ddp = int(os.environ.get('RANK', -1)) != -1 #is ddp running?
if ddp:
	assert cuda.is_avaiable(), 'make sure you have access to gpu!'
	init_process_group.backend('nccl') #nccl for communication btween GPU-GPU
	ddp_rank = int(os.environ('RANK')) #which gpu? 
	ddp_local_rank= int(os.environ('LOCAL_RANK')) #gpu on specific node, e.g. gpu 1 on node 0, gpu 0 on node 1
	ddp_world_size=int(os.environ('WORLD_SIZE')) #total numb of process running 
	device = f'cuda:{ddp_local_rank}'
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0 #the very first process will do the extra work of logging, checkpointing, etc
else: #if not multi-gpu then revert back to single gpu set up
	ddp_rank = 0
	ddp_local_rank = 0
	ddp_world_size = 1
	master_process = True
	device = 'cpu'
	if torch.cuda.is_avaiable():
		device = 'cuda'
```

## How this multi-gpu set up can affect the rest of the code for training? 
When it comes to multi-gpu setup you should always think of there are 8 process that are happening at the same time, so 8 jobs that are reading the code exactly same way and roughly speaking unaware of the adjacent process, right? so if you imagine this way then let's think thro which part of the code has to change respectively;
So as an exp, imagine we wanna simulate a `total_batch_size = 524288` the very first things we need to be careful abt are 
- `grad_accum_steps` ---> which now given 8 gpus in parallel will be different;

```python
total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure your total desired batch side is divisible by B*T*ddp_world_size'
grad_accum_size = total_batch_size // (B * T * ddp_world_size) #2**19 / (16*1024*8 = 131k in each fwrd/bwrd) = 4
if master_process:
	print(f'total batch size:{total_batch_size}')
	print(f'total grad accum steps:{grad_accum_size}')

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
```

Nxt we obviously need to refactor the dataloader to reflect the current setup;
- `data_loader` to reflect the multi-process setup to make sure that each process is processing a different part of the data!
```python
class DataLoaderLite:
	def __init__(self, B, T, process_rank, num_processes):
		self.B = B
		self.T = T
		self.ddp_rank = ddp_rank
		self.num_processes = num_processes
		#very tiny dataset that at init we load it from disk n save it in the memory 
		with open('input.txt', 'r') as f:
			text = f.read()
		enc = tiktoken.get_encoding('gpt2')
		tokens = enc.encode(text)
		self.tokens = torch.tensor(tokens)
		print(len(self.tokens))
		# so process_rank = 0 will start at zero but process_rank=1, gotta start by B*T then next one 2*(B*T)
		self.current_position = self.B * self.T * self.processes_rank # used to be 0, as we are starting at the beginning

	def next_batch(self):
		B, T = self.B, self.T
		buffer = self.tokens[self.current_position: self.current_position+B*T+1]
		x = (buffer[:-1]).view(B,T) #inputs
		y = (buffer[1:]).view(B,T) #target
		#advance the positionin the tensor
		self.current_position += B*T*self.num_processes
		#if its the end of the dataset; reset by the entire chunk
		if self.current_position + (B*T*self.num_processes +1) > len(self.tokens):
			self.current_position = self.B * self.T * self.processes_rank
			
		return x,y		
```

IDENTICAL:
```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank 
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(len(self.tokens))

        # Each GPU starts at a different initial position
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buffer[:-1].view(B, T)  # inputs
        y = buffer[1:].view(B, T)   # targets

        # Advance position by number of tokens consumed across all processes
        self.current_position += B * T * self.num_processes

        # Check if the next position exceeds the dataset length
        if self.current_position + (B * T + 1) > len(self.tokens):
            # Reset back to initial position unique for this GPU/process
            self.current_position = self.B * self.T * self.process_rank

        return x, y
```

So now all the gpus will come here n copy the model; and they all compile the model, 
## How abt modelling phase?
note that we are currently working within the step so NONE of the lr stuff won't change!
```python
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
model = torch.compile(model)
if ddp:
	model = DDP(model, device_ids=[local_ddp_rank])
```
here when we wrap the model in `DDP` essentially what will happen is;
- in the forward pass there will not be any change n its gonna be identical
- but in backward pass, so during the backward pass the gradients are calculated on each gpu on each node, now what this DDP does is to call `allreduce` to deposit the `average` of all computed gradients on each gpu, therefore, at the end of the bcwd pass , each of the gpus on each node will have the avrg of all the gradients on them and this is essentially the  syncronization of the gradients between gpus!
```python
lr = .5e-10
optimizer = torch.optim.AdamW()

for step in range(max_steps):
	optimizer.zero_grad()
	loss_accum = 0
	for micro_step in range(grad_accum_steps):
		x,y = train_loader.next_batch()
		x,y = x.to(device), y.to(device)
	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits, loss = model(x,y)
	loss = loss / grad_accum_steps
	loss_accum += loss.detach()
	loss.backward()
```
but the problem here is bc of the grad_accum_steps, the loss is just depositing each time and they are adding up together, but we don't want to syncronize each time that the loss is deposited, instead we wanna only syncronize when the `grad_accum_steps` are over, right?
so the above code will change to this:

```python
lr = .5e-10
optimizer = torch.optim.AdamW()

for step in range(max_steps):
	optimizer.zero_grad()
	loss_accum = 0
	for micro_step in range(grad_accum_steps):
		x,y = train_loader.next_batch()
		x,y = x.to(device), y.to(device)
	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits, loss = model(x,y)
	loss = loss / grad_accum_steps
	loss_accum += loss.detach()
	if ddp:
		self.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
	loss.backward()
	if ddp:
		dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
```
so when this is done, every single rank will magically have the average of all the gradinets on them, same goes for the loss, we want the average of the loss also be abaiable on each of the ranks!
and finally;
```python
total_token_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
#finally exit from each process
if ddp:
	destroy_process_group()
```