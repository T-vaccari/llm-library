
import math

def cosine_lr_scheduler(t, alpha_max, alpha_min, T_w, T_c):
   if t < T_w:
      return (t/T_w) * alpha_max
   if T_w<= t <= T_c:
      return  alpha_min + 0.5 * (1 + math.cos(((t- T_w)/(T_c -T_w)) * math.pi)) * (alpha_max-alpha_min)
   else:
      return alpha_min