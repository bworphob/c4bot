import numpy as np
import collections
import copy
import math
import random


CPUCT = 1.0        
SEARCH_LOOP = 500  

class DummyNode:
    """โหนดจำลองที่ทำหน้าที่เป็นจุดเริ่มต้นสูงสุด เพื่อให้ Root Node จริงมี Parent"""
    def __init__(self):
        self.parent = None
        self.child_value = collections.defaultdict(float)
        self.child_num_visit = collections.defaultdict(float)

class MCNode:
    """โหนดหลักของต้นไม้ MCTS"""
    def __init__(self, game, move=None, parent=None):
        self.game = game        # สถานะกระดานในโหนดนี้
        self.move = move        # ท่าเดินที่ทำให้เกิดโหนดนี้
        self.parent = parent    # โหนดแม่
        self.children = {}      # โหนดลูก (เก็บแบบ Dictionary {move: MCNode})
        
        # ข้อมูลสำหรับคำนวณ UCB
        self.child_prob = np.zeros(7, dtype=np.float32)      # สัญชาตญาณจากสมอง (P)
        self.child_value = np.zeros(7, dtype=np.float32)     # คะแนนสะสม (W)
        self.child_num_visit = np.zeros(7, dtype=np.float32) # จำนวนครั้งที่ถูกสำรวจ (N)
        
        self.is_expanded = False # เช็คว่าโหนดนี้ถูกแตกกิ่งหรือยัง

    @property # ทำให้ function ดูเหมือนตัวแป
    def num_visit(self):
        return self.parent.child_num_visit[self.move]

    @num_visit.setter # ทำให้ assign ค่าได้เหมือนตัวแปรปกติ
    def num_visit(self, value):
        self.parent.child_num_visit[self.move] = value

    @property
    def value(self):
        return self.parent.child_value[self.move]

    @value.setter
    def value(self, value):
        self.parent.child_value[self.move] = value

    def get_ucb_scores(self):
        """คำนวณคะแนนความน่าสนใจของลูกๆ ทุกตัว (Q + U)"""
        # Q = ค่าเฉลี่ยคะแนนจากประสบการณ์
        Q = self.child_value / (1 + self.child_num_visit)
        # U = คะแนนโบนัสจากการสำรวจตามสัญชาตญาณ
        U = CPUCT * self.child_prob * (math.sqrt(self.num_visit) / (1 + self.child_num_visit))
        return Q + U

    def select_best_move(self):
        """เลือกท่าเดินที่ดีที่สุด (พิจารณาเฉพาะช่องที่หยอดได้จริง)"""
        valid_actions = self.game.validAction()
        ucb_scores = self.get_ucb_scores()
        
        # เลือก index ที่ให้ค่า ucb สูงสุดเฉพาะในช่องที่ valid
        best_move = valid_actions[np.argmax(ucb_scores[valid_actions])]
        return best_move

    def expand(self, action_probs):
        """แตกกิ่งก้านตามความน่าจะเป็นที่สมองให้มา"""
        valid_actions = self.game.validAction()
        self.is_expanded = True
        
        # เก็บความน่าจะเป็นเฉพาะช่องที่เดินได้จริง ช่องที่เดินไม่ได้ให้เป็น 0
        new_probs = np.zeros(7, dtype=np.float32)
        for move in valid_actions:
            new_probs[move] = action_probs[move]
            
        # ถ้าเป็นโหนดราก (Root) ให้ใส่ Noise เพื่อความหลากหลาย (แบบ AlphaZero)
        if isinstance(self.parent, DummyNode):
            noise = np.random.dirichlet([0.3] * len(valid_actions))
            new_probs[valid_actions] = 0.75 * new_probs[valid_actions] + 0.25 * noise
            
        self.child_prob = new_probs

class ZeroPlayer:
    def __init__(self, brain):
        self.brain = brain

    # def MCTS(self, game):
    #     """กระบวนการคิดล่วงหน้า 500 รอบ"""
    #     root = MCNode(game, parent=DummyNode())

    #     for _ in range(SEARCH_LOOP):
    #         # 1. Selection: เดินลงไปที่ใบ (Leaf) ของต้นไม้ที่น่าสนใจที่สุด
    #         node = root
    #         while node.is_expanded:
    #             move = node.select_best_move()
    #             # ถ้ากิ่งนี้ยังไม่มีในระบบ ให้สร้างขึ้นมา
    #             if move not in node.children:
    #                 next_game = copy.deepcopy(node.game)
    #                 next_game.insertColumn(move)
    #                 node.children[move] = MCNode(next_game, move=move, parent=node)
    #             node = node.children[move]

    #         # 2. Evaluation & Expansion: ถามสมองและแตกกิ่ง
    #         policy, value = self.brain.predict(node.game.getStateAsPlayer())
            
    #         if not node.game.isEnd:
    #             node.expand(policy)
    #         else:
    #             # กำหนดคะแนนตามผลแพ้ชนะจริง
    #             if node.game.winner == 0: value = 0
    #             else: value = 1 if node.game.winner == 1 else -1

    #         # 3. Backpropagation: อัปเดตคะแนนย้อนกลับขึ้นไปถึงจุดเริ่มต้น
    #         while node.parent is not None:
    #             node.num_visit += 1
    #             # สลับเครื่องหมายตามมุมมองผู้เล่น (เหมือนรุ่นพี่)
    #             # ถ้า parent ของเรามี parent อีกที แปลว่า parent เราไม่ใช่ DummyNode
    #             if node.parent.parent is not None:
    #                 if node.parent.game.current_turn == 2:
    #                     node.value += value
    #                 else:
    #                     node.value -= value
                        
    #             node = node.parent
    #     return root
    def MCTS(self, game):
        """กระบวนการคิดล่วงหน้า 500 รอบ"""
        root = MCNode(game, parent=DummyNode())

        for _ in range(SEARCH_LOOP):
            # 1. Selection
            node = root
            while node.is_expanded:
                move = node.select_best_move()
                if move not in node.children:
                    next_game = copy.deepcopy(node.game)
                    next_game.insertColumn(move)
                    node.children[move] = MCNode(next_game, move=move, parent=node)
                node = node.children[move]

            # 2. Evaluation & Expansion
            policy, value = self.brain.predict(node.game.getStateAsPlayer())
            
            if not node.game.isEnd:
                node.expand(policy)
            else:
                # กำหนดคะแนนตามผลแพ้ชนะจริง (Relative: self=1, opponent=-1)
                if node.game.winner == 0: 
                    value = 0
                else: 
                    # ถ้าผู้ชนะตรงกับ current_turn ให้ 1, ไม่ตรงให้ -1
                    value = 1 if node.game.winner == node.game.current_turn else -1
                    # # ถ้าคนชนะคือผู้เล่น 1 ให้ค่าเป็น 1, ถ้าผู้เล่น 2 ชนะให้ค่าเป็น -1
                    # value = 1 if node.game.winner == 1 else -1


            # 3. Backpropagation
            # ส่งค่า Value ย้อนกลับไปยัง Root โดยสลับสัญญาณก่อนนำไปเก็บ
            # เพราะ value ถูกกำหนดจากมุมมองของผู้เล่นที่อยู่ใน leaf/node นั้น
            temp_node = node
            current_value = -value
            while temp_node.parent is not None:
                temp_node.num_visit += 1
                temp_node.value += current_value
                current_value = -current_value
                temp_node = temp_node.parent
        return root


    def act(self, game, tau=0):
        """เลือกท่าเดินจริงบนกระดาน"""
        # รัน MCTS เพื่อสร้างต้นไม้ความคิด
        root = self.MCTS(copy.deepcopy(game))
        
        # คำนวณความมั่นใจ (Policy) จากจำนวนครั้งที่ไปสำรวจ (Visit Count)
        visit_counts = root.child_num_visit
        policy = visit_counts / np.sum(visit_counts)

        # ถ้า tau=1 (ตอนเทรน) จะสุ่มเดินตามความมั่นใจ, ถ้า tau=0 (แข่งจริง) จะเลือกช่องที่มั่นใจที่สุด
        if tau == 1:
            action = np.random.choice(7, p=policy)
        else:
            action = np.argmax(policy)
            
        return action, policy
        






class TRTPlayer(ZeroPlayer):
    def __init__(self, trt_brain, n_simulations=400):
        # trt_brain คืออินสแตนซ์ของ TRTBrainWrapper
        self.brain = trt_brain
        self.n_simulations = n_simulations

    def MCTS(self, game):
        """ กระบวนการคิดล่วงหน้าโดยใช้ TensorRT """
        root = MCNode(game, parent=DummyNode())

        for _ in range(self.n_simulations):
            # 1. Selection
            node = root
            while node.is_expanded:
                move = node.select_best_move()
                if move not in node.children:
                    next_game = copy.deepcopy(node.game)
                    next_game.insertColumn(move)
                    node.children[move] = MCNode(next_game, move=move, parent=node)
                node = node.children[move]

            # 2. Evaluation & Expansion (ใช้ TRT Brain)
            # สำคัญ: trt_brain.predict() คืนค่า (policy, value)
            policy, value = self.brain.predict(node.game)
            
            if not node.game.isEnd:
                node.expand(policy)
            else:
                if node.game.winner == 0: 
                    value = 0
                else: 
                    value = 1 if node.game.winner == node.game.current_turn else -1

            # 3. Backpropagation
            # ส่งค่า Value ย้อนกลับไปยัง Root โดยสลับสัญญาณก่อนนำไปเก็บ
            # เพราะ value ถูกกำหนดจากมุมมองของผู้เล่นที่อยู่ใน leaf/node นั้น
            temp_node = node
            current_value = -value
            while temp_node.parent is not None:
                temp_node.num_visit += 1
                temp_node.value += current_value
                current_value = -current_value
                temp_node = temp_node.parent
        return root