import torch
import math
from einops import rearrange, repeat
from .dynamic_attn_triton import FlashAttnFunc

def dyn_attention_ref_multi(
    q_list,
    k_list,
    v_list,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
    return_score_list=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q_list[0].dtype
    if upcast:
        for i in range(len(q_list)):
            q_list[i] = q_list[i].float()
            k_list[i] = k_list[i].float()
            v_list[i] = v_list[i].float()
        
    d = q_list[0].shape[-1]
    if not reorder_ops:
        score_list = []
        for i in range(len(q_list)):
            scores = torch.einsum("bthd,bshd->bhts", q_list[i] / math.sqrt(d), k_list[i])
            score_list.append(scores)
    else:
        score_list = []
        for i in range(len(q_list)):
            scores = torch.einsum("bthd,bshd->bhts", q_list[i], k_list[i] / math.sqrt(d))
            score_list.append(scores)
        # scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
        # scores1 = torch.einsum("bthd,bshd->bhts", q1, k1 / math.sqrt(d))
    scores = torch.cat(score_list, dim=-1)
    # scores = torch.cat([scores, scores1], dim=-1)
    # v = torch.cat([v, v1], dim=1)
    v = torch.cat(v_list, dim=1)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    
    if return_score_list:
        return output.to(dtype=dtype_og), attention.to(dtype=dtype_og), score_list
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


class InplaceModify(torch.autograd.Function):
    """
    自定义 autograd Function 实现原地修改并保持梯度流
    前向传播：应用修改后的值
    反向传播：使用修改后的值计算梯度
    """
    @staticmethod
    def forward(ctx, input_tensor, modified_value):
        # inplcae modify tensor
        input_tensor.data.copy_(modified_value)
        # 必须返回一个新的 tensor 引用以保持计算图
        return input_tensor.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时直接返回上游梯度
        # 表示"修改操作"本身的导数为1（恒等变换）
        return grad_output, None

inplace_modify = InplaceModify.apply

flash_att = FlashAttnFunc.apply

def dyn_attention_ref(
    q,
    k,
    v,
    q1,
    k1,
    v1,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        q1, k1, v1 = q1.float(), k1.float(), v1.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    k1 = repeat(k1, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v1 = repeat(v1, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
        scores1 = torch.einsum("bthd,bshd->bhts", q1 / math.sqrt(d), k1)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
        scores1 = torch.einsum("bthd,bshd->bhts", q1, k1 / math.sqrt(d))
    
    scores = torch.cat([scores, scores1], dim=-1)
    v = torch.cat([v, v1], dim=1)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def dyn_attn(q, k, v, q1, k1, v1):
    # b, seq_len, n_head, head_dim
    # out0 = flash_att(q, k, v, None, False, None)
    # m0 = out0.grad_fn.m
    # lse0_full = out0.grad_fn.lse
    out0, m0, lse0_full = flash_att(q, k, v, None, False, None)

    # out1 = flash_att(q1, k1, v1, None, False, None)
    # m1 = out1.grad_fn.m
    # lse1_full = out1.grad_fn.lse
    out1, m1, lse1_full = flash_att(q1, k1, v1, None, False, None)

    with torch.no_grad():
        lse0 = lse0_full[:, :, :out0.shape[1]].detach()
        m0 = m0[:, :, :out0.shape[1]].detach()
        # lse = m + torch.log(l)
        l0 = torch.exp(lse0 - m0)

        lse1 = lse1_full[:, :, :out1.shape[1]].detach()
        m1 = m1[:, :, :out1.shape[1]].detach()
        # lse = m + torch.log(l)
        l1 = torch.exp(lse1 - m1) # b, nh, seq_len_q

        m_new = torch.where(m0 > m1, m0, m1)
        l_new = torch.exp(lse0 - m_new) + torch.exp(lse1 - m_new)
        # l_new = l0 * torch.exp(m0 - m_new) + l1 * torch.exp(m1 - m_new)

        lse_new = m_new + torch.log(l_new)

        # b, nh, seq, c
        scale0 = torch.exp(m0 - m_new)
        out0_new = out0.permute(0, 2, 1, 3).contiguous() * l0.unsqueeze(-1) * scale0.unsqueeze(-1)

        scale1 = torch.exp(m1 - m_new)
        out1_new = out1.permute(0, 2, 1, 3).contiguous() * l1.unsqueeze(-1) * scale1.unsqueeze(-1)

        out0_new = out0_new / l_new.unsqueeze(-1)
        out0_new = out0_new.permute(0, 2, 1, 3).contiguous()

        out1_new = out1_new / l_new.unsqueeze(-1)
        out1_new = out1_new.permute(0, 2, 1, 3).contiguous() # b, nh, seq, c

    lse0_full[:, :, :out0.shape[1]] = lse_new 
    lse1_full[:, :, :out1.shape[1]] = lse_new

    if out0.grad_fn is not None:
        print("====checkpoint backward forwarding====")
        with torch.no_grad():
            out0.grad_fn.lse = lse0_full
            # out0.grad_fn.o = out0_new
            out0.grad_fn.o = (out0_new + out1_new).detach()

            out1.grad_fn.lse = lse1_full
            # out1.grad_fn.o = out1_new
            out1.grad_fn.o = (out0_new + out1_new).detach()
    else:
        print("====check point forawding=====")

    # then replace output value of each attention
    out0 = inplace_modify(out0, out0_new)
    out1 = inplace_modify(out1, out1_new)

    out = out0 + out1

    return out

def dyn_attn_multi(q_list, k_list, v_list):
    # b, seq_len, n_head, head_dim
    out_list = []
    m_list = []
    lse_list = []
    m_max = None

    # block-wise forward
    for q, k, v in zip(q_list, k_list, v_list):
        out, m, lse = flash_att(q, k, v, None, False, None)
        # m = out.grad_fn.m
        with torch.no_grad():
            if m_max is None:
                m_max = m
            else:
                m_max = torch.where(m > m_max, m, m_max)
        # lse = out.grad_fn.lse
        out_list.append(out)
        m_list.append(m)
        lse_list.append(lse)
    
    with torch.no_grad():
        # compute l_new
        l_new = 0
        for i in range(len(out_list)):
            lse = lse_list[i]
            l_new = l_new + torch.exp(lse - m_max)
        
        lse_new = m_max + torch.log(l_new)
        
        # compute online softmax out
        out_final = 0.
        out_new_list = []
        for i in range(len(out_list)):
            out = out_list[i]
            lse = lse_list[i]
            m = m_list[i]
            l = torch.exp(lse - m)[:, :, :out.shape[1]]
            scale = torch.exp(m - m_max)[:, :, :out.shape[1]]
            out_new = out.permute(0, 2, 1, 3).contiguous() * l.unsqueeze(-1) * scale.unsqueeze(-1)
            out_new = out_new / l_new[:, :, :out.shape[1]].unsqueeze(-1)
            out_new = out_new.permute(0, 2, 1, 3).contiguous()
            out_new_list.append(out_new)
            out_final = out_final + out_new
        
        if out_list[0].grad_fn is not None:
            print("====checkpoint backwward forwaring multi!!!====")
            # replace lse for gradient computing
            for i in range(len(out_list)):
                out_list[i].grad_fn.lse = lse_new
                # out0.grad_fn.o = out0_new
                out_list[i].grad_fn.o = out_final.detach()
        else:
            print("====checkpoint forwaring multi!!!====")
    
    # need gradients
    out = 0.
    for i in range(len(out_new_list)):
        x = inplace_modify(out_list[i], out_new_list[i])
        out = out + x
    
    return out