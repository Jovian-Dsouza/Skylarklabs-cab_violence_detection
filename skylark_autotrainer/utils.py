from pytorch_lightning import LightningModule

def get_model_memory(model: LightningModule):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return str(mem / 1e+6) + 'MB'

def commas(views):
	ans = []
	for i, num in enumerate(''.join(reversed(str(views)))):
		ans.append(num)
		if (i + 1) % 3 == 0 and i + 1 != len(str(views)):
			ans.append(",")
	return ''.join(reversed(ans))

def get_model_parameters_count(model: LightningModule):
    return commas(sum([param.numel() for param in model.parameters()]))