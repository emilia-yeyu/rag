-- 迎宾机器人知识库数据库表结构
-- 人物信息表
CREATE TABLE IF NOT EXISTS person_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    title TEXT,
    department TEXT,
    office_location TEXT,
    phone TEXT,
    email TEXT,
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

-- 插入示例数据
INSERT OR IGNORE INTO person_info (name, title, department, office_location, phone, email, bio) VALUES
('张三', '技术总监', '技术部', 'A栋3楼301室', '13800138001', 'zhangsan@company.com', '负责公司技术战略规划和团队管理，有10年技术管理经验'),
('李四', '产品经理', '产品部', 'B栋2楼205室', '13800138002', 'lisi@company.com', '专注于用户体验设计，主导过多个重要产品项目'),
('王五', '销售总监', '销售部', 'C栋1楼101室', '13800138003', 'wangwu@company.com', '带领销售团队，年销售额超过5000万'),
('赵六', '人事经理', '人事部', 'A栋2楼201室', '13800138004', 'zhaoliu@company.com', '负责公司人才招聘和员工关系管理');

-- 插入示例会议数据
INSERT OR IGNORE INTO meeting_schedule (person_id, meeting_title, meeting_date, start_time, end_time, location, attendees, description) VALUES
(1, '技术团队周会', '2024-01-15', '09:00:00', '10:00:00', '会议室A', '技术部全体成员', '讨论本周技术进展和下周计划'),
(2, '产品评审会', '2024-01-15', '14:00:00', '16:00:00', '会议室B', '产品部、技术部', '新功能产品评审'),
(3, '客户拜访', '2024-01-16', '10:00:00', '11:30:00', '客户公司', '重要客户', '季度业务回顾'),
(4, '员工培训', '2024-01-16', '15:00:00', '17:00:00', '培训室', '新员工', '企业文化培训'); 