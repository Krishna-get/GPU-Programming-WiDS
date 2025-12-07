
# **Week 1 — GPU Intuition & Compute Foundations**

Welcome to Week 1.
This week is all about building **intuition** — understanding *why* GPUs exist, *how* they differ from CPUs, and *when* GPU acceleration makes sense for scientific and AI workloads.

This is the conceptual foundation for the entire course.
You will not write CUDA kernels yet — that starts in Week 2.



##  **Learning Goals**

By the end of this week, you should be able to:

* Explain the architectural differences between CPUs and GPUs
* Understand SIMT parallelism, warps, and throughput computing
* Identify workloads that benefit from GPU acceleration
* Understand the CUDA execution hierarchy (threads → blocks → grids)
* Identify at least one potential GPU-accelerable task from your own research or interests

This week builds your mental model so everything later (memory, performance, kernels) fits together.



## 📘 **Required Resources**

### **1. NVIDIA CUDA Programming Guide (Conceptual Introduction)**

Read: **Chapter 1 — Introduction**
[https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)

Focus on:

* What GPUs are optimized for
* Parallel execution model
* Overview of CUDA threads/blocks/grids



### **2. UC Berkeley CS61C — Parallel Processors (Lecture 17)**

Video:
[https://www.youtube.com/watch?v=gychBEOgG8A](https://www.youtube.com/watch?v=gychBEOgG8A)
(If link breaks, search: *CS61C Parallel Processors Lecture 17*)

This gives the “physics layer” of performance:

* Why CPUs stopped getting faster
* Why GPUs thrive on massive parallelism
* Memory bottlenecks and throughput considerations



### **3. Stanford CS231n — Hardware/Software Interface (Lecture 15)**

Slides & Video:
[https://www.youtube.com/watch?v=WGf1f2HbJpE](https://www.youtube.com/watch?v=WGf1f2HbJpE)

Focus on:

* GPU in the modern AI stack
* How frameworks like PyTorch map operations to hardware



## 🧠 **Concepts Covered This Week**

* CPU vs GPU architecture
* SIMD vs SIMT
* Warps, streaming multiprocessors, and massive parallelism
* Throughput computing vs latency computing
* CUDA execution hierarchy
* Identifying GPU-friendly workloads
* Profiling intuition: *what part of your code is slow? why?*



## 🛠 **Environment Setup Reminder**

You can use:

### **Option A — Your own laptop with NVIDIA GPU**

(Install CUDA Toolkit, drivers, PyTorch CUDA — see main README.)

### **Option B — Google Colab**

(Enable GPU: Runtime → Change runtime type → GPU)

No coding is required this week, but you should verify that your environment can detect a GPU.

Example (run this in Colab or terminal):

```python
!nvidia-smi
```



## 🧪 **Week 1 Assignment (Summary)**

Your full assignment instructions are in `assignment.md`.
Here is a preview of what you will do:

### **Task 1 — Identify a GPU-Accelerable Problem**

Pick a workload you care about:

* Research simulation
* Numerical solver
* ML operation / training bottleneck
* Data analysis pipeline
* Anything computation-heavy in Python

Write a short description of **why** it may benefit from GPU acceleration.



### **Task 2 — GPU Execution Model Diagram**

Draw (on paper or digitally) the CUDA hierarchy:

```
Grid
 └── Blocks
       └── Threads
```

Annotate each level and briefly describe how your chosen workload might map to it.



### **Task 3 — Reflection**

In 150–300 words, explain:

* What you learned from the lectures
* What types of tasks are GPU-friendly
* What surprised you the most about GPU architecture



**Submission folder:**
Place your answers in the `week1` folder:

```
week1/
 ├── assignment.md
 ├── diagrams/ (if any)
 └── notes.md (optional)
```



## 📚 **Optional (Highly Recommended) Extra Resources**

* “Even Easier Introduction to CUDA” — NVIDIA Blog
  [https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* “How GPUs Work” — Stanford CME193 notes
* Chapters 1 & 2 of *GPU Gems 3*



If this looks good, I’ll now generate:

👉 **`week1/assignment.md`** with fully detailed instructions, templates, and submission format.
