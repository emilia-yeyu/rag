-- AMICRO一微半导体迎宾机器人知识库数据库表结构
-- 人物信息表
CREATE TABLE IF NOT EXISTS person_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    title TEXT,
    department TEXT,
    office_location TEXT,
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 会议日程表
CREATE TABLE IF NOT EXISTS meeting_schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    meeting_title TEXT NOT NULL,
    meeting_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    location TEXT,
    attendees TEXT,
    description TEXT,
    status TEXT DEFAULT 'scheduled', -- scheduled, in_progress, completed, cancelled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES person_info(id) ON DELETE CASCADE
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_person_name ON person_info(name);
CREATE INDEX IF NOT EXISTS idx_meeting_person ON meeting_schedule(person_id);
CREATE INDEX IF NOT EXISTS idx_meeting_date ON meeting_schedule(meeting_date);

-- 插入AMICRO一微半导体实际人员数据
INSERT OR IGNORE INTO person_info (name, title, department, office_location, bio) VALUES
-- 创始团队与高管
('肖刚军', '创始人兼总经理', '总经办', '珠海总部15楼总经办', '公司创始人及总经理，负责公司整体战略规划和运营管理'),
('姜新桥', '联合创始人兼总裁', '总经办', '珠海总部15楼总经办', '联合创始人及总裁，负责公司重大战略决策和业务发展'),
('许登科', '联合创始人兼副总经理', '总经办', '珠海总部15楼总经办', '联合创始人及副总经理，参与公司核心业务决策'),
('黄明强', '副总经理兼制造中心负责人', '制造中心', '珠海总部14楼制造中心', '副总经理及制造中心负责人，负责采购、仓管、物流、报关和质量管理'),
('杨武', '市场营销一中心总经理', '营销一中心', '珠海总部营销中心', '市场营销一中心总经理，负责家用清洁机器人业务'),
('赖钦伟', '系统研发中心高级总监', '系统研发中心', '珠海总部15楼系统研发中心', '系统研发中心高级总监，负责系统规格定义、设计、验证和固件开发'),

-- 研发团队
('周和文', '机器人研发总监', '机器人研发中心', '珠海总部14楼机器人研发中心', '机器人研发总监，专注于机器人技术的研发与交付，涵盖感知系统、多传感器融合决策等'),
('李永勇', '机器人研发副总监', '机器人研发中心', '珠海总部14楼机器人研发中心', '机器人研发副总监，负责机器人核心技术开发'),
('唐伟华', '机器人研发副总监', '机器人研发中心', '珠海总部14楼机器人研发中心', '机器人研发副总监，专注于机器人技术创新'),
('赵伟兵', 'IC研发总监', 'IC研发中心', '珠海总部15楼IC研发中心', 'IC研发总监，负责IC电路的规格定义、设计、验证以及芯片物理实现'),
('成世明', 'IC研发项目总监', 'IC研发中心', '珠海总部15楼IC研发中心', 'IC研发项目总监，负责IC研发项目管理和技术协调'),
('何再生', 'IC研发副总监', 'IC研发中心', '珠海总部15楼IC研发中心', 'IC研发副总监，专注于集成电路设计和验证'),
('常子奇', 'IC研发技术总监', 'IC研发中心', '珠海总部15楼IC研发中心', 'IC研发技术总监，负责IC技术架构和方案设计'),
('钟伟金', '系统设计负责人', '系统研发中心', '珠海总部15楼系统研发中心', '系统设计负责人，专注于系统架构设计和技术方案'),
('林立', '系统设计负责人', '系统研发中心', '珠海总部15楼系统研发中心', '系统设计负责人，负责硬件应用方案的开发与测试');

-- 插入AMICRO实际会议数据
INSERT OR IGNORE INTO meeting_schedule (person_id, meeting_title, meeting_date, start_time, end_time, location, attendees, description) VALUES
-- 高管会议
(1, '公司月度经营会', '2024-12-15', '09:00:00', '11:00:00', '特斯拉会议室', '总经办全体成员', '讨论公司月度经营状况和下月计划'),
(2, '战略规划讨论会', '2024-12-15', '14:00:00', '16:00:00', '爱因斯坦会议室', '创始团队、各中心总监', '2025年公司战略规划讨论'),

-- 研发会议
(7, '机器人技术评审会', '2024-12-16', '09:00:00', '11:00:00', '法拉第会议室', '机器人研发中心团队', '机器人感知系统技术方案评审'),
(8, '机器人研发周例会', '2024-12-16', '14:00:00', '15:30:00', '黎曼会议室', '机器人研发团队', '讨论本周研发进展和技术难点'),
(10, 'IC设计方案评审', '2024-12-17', '10:00:00', '12:00:00', '爱迪生会议室', 'IC研发中心团队', '新款芯片设计方案技术评审'),
(11, 'IC项目进度同步会', '2024-12-17', '15:00:00', '16:00:00', '诺贝尔会议室', 'IC研发项目组', '芯片项目进度同步和问题讨论'),
(6, '系统架构设计会议', '2024-12-18', '09:00:00', '11:00:00', '牛顿会议室', '系统研发中心团队', '系统架构设计方案讨论'),

-- 业务会议
(5, '营销策略讨论会', '2024-12-18', '14:00:00', '16:00:00', '剑桥会议室', '营销中心团队', '家用清洁机器人市场策略讨论'),
(4, '制造中心周例会', '2024-12-19', '09:00:00', '10:00:00', '摩尔会议室', '制造中心团队', '生产计划和质量管理讨论'),

-- 洽谈会议
(1, '投资机构路演', '2024-12-19', '14:00:00', '16:00:00', '追求空间', '投资机构代表', '公司业务发展情况介绍和融资洽谈'),
(2, '合作伙伴交流会', '2024-12-20', '10:00:00', '12:00:00', '卓越空间', '重要合作伙伴', '技术合作和业务拓展讨论'); 