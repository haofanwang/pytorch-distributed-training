# pytorch-distributed-training
A simple cookbook for DDP training in Pytorch.

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
