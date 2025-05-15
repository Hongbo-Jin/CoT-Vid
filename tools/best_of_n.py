


def evaluate_candidates(
    candidates,
    question,
    score_model
):
    for candidate in candidates:
        
        features = {}
        # 文本预处理
        q_doc = score_model(question)
        s_doc = score_model(candidate)
    
        # 特征1：语义相关性（双向相似度）
        q_vector = q_doc.vector.reshape(1, -1)
        s_vector = s_doc.vector.reshape(1, -1)
        features['semantic_sim'] = (cosine_similarity(q_vector, s_vector)[0][0] + 1) / 2  # 归一化到0-1
        
        scores.append(features['semantic_sim'])
        
    scores=[]
    return scores

