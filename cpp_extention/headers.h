#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)


#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000