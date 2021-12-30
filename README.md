# pytorch-distributed-training
A simple cookbook for DDP training in Pytorch. To specify the number of GPU per node, you can change the ```nproc_per_node``` and ```CUDA_VISIBLE_DEVICES``` defined in ```train.sh```.

You can find your ID address via

```bash
import socket

print('The name of machine is: ' + socket.gethostname())
 
print('The IP address is: ' + socket.gethostbyname(socket.gethostname()))
```

# Single machine
```bash
$ sh train.sh 0 1
```

# Multiple machines

On machine 1
```bash
$ sh train.sh 0 2
```

On machine 2
```bash
$ sh train.sh 1 2
```

The order of each command does not matter. The training only gets started when all commands are started.

# Notes
1. Ensure proper communication across multiple machines, or the program will hang out in DDP constructor.
2. Ensure the version of Pytorch is the same across each machine, or it may cast unpredictable errors.
