-- AMICRO一微半导体迎宾机器人数据库 - 增删改查操作示例
-- 使用方法：复制需要的SQL语句，修改参数后执行

-- ========================================
-- 查询操作 (SELECT)
-- ========================================

-- 1. 查看所有人员信息
SELECT * FROM person_info ORDER BY name;

-- 2. 查看所有会议信息（包含负责人姓名）
SELECT 
    m.id,
    m.meeting_title,
    p.name as person_name,
    p.title as person_title,
    m.meeting_date,
    m.start_time,
    m.end_time,
    m.location,
    m.attendees,
    m.description,
    m.status
FROM meeting_schedule m
JOIN person_info p ON m.person_id = p.id
ORDER BY m.meeting_date DESC, m.start_time ASC;

-- 3. 按姓名查询特定人员
SELECT * FROM person_info WHERE name = '肖刚军';

-- 4. 模糊搜索人员（按姓名）
SELECT * FROM person_info WHERE name LIKE '%肖%';

-- 5. 按部门查询人员
SELECT * FROM person_info WHERE department = '总经办';

-- 6. 查询特定人员的所有会议
SELECT 
    p.name,
    p.title,
    m.meeting_title,
    m.meeting_date,
    m.start_time,
    m.end_time,
    m.location
FROM person_info p
JOIN meeting_schedule m ON p.id = m.person_id
WHERE p.name = '肖刚军'
ORDER BY m.meeting_date DESC;

-- 7. 查询今天的会议
SELECT 
    m.meeting_title,
    p.name as person_name,
    m.start_time,
    m.end_time,
    m.location,
    m.attendees
FROM meeting_schedule m
JOIN person_info p ON m.person_id = p.id
WHERE m.meeting_date = DATE('now')
ORDER BY m.start_time;

-- 8. 查询指定日期的会议
SELECT 
    m.meeting_title,
    p.name as person_name,
    m.start_time,
    m.end_time,
    m.location,
    m.attendees
FROM meeting_schedule m
JOIN person_info p ON m.person_id = p.id
WHERE m.meeting_date = '2024-12-15'
ORDER BY m.start_time;

-- 9. 查询本周的会议
SELECT 
    m.meeting_title,
    p.name as person_name,
    m.meeting_date,
    m.start_time,
    m.location
FROM meeting_schedule m
JOIN person_info p ON m.person_id = p.id
WHERE m.meeting_date BETWEEN DATE('now', 'weekday 0', '-7 days') 
    AND DATE('now', 'weekday 0')
ORDER BY m.meeting_date, m.start_time;

-- 10. 统计各部门人数
SELECT department, COUNT(*) as person_count 
FROM person_info 
GROUP BY department 
ORDER BY person_count DESC;

-- ========================================
-- 插入操作 (INSERT)
-- ========================================

-- 11. 添加新人员（示例）
-- 注意：修改下面的信息为实际数据
INSERT INTO person_info (name, title, department, office_location, bio) VALUES 
('张三', '软件工程师', 'IC研发中心', '珠海总部15楼IC研发中心', '专注于芯片软件开发，有5年相关经验');

-- 12. 添加新会议（示例）
-- 注意：person_id需要是已存在的人员ID
INSERT INTO meeting_schedule (person_id, meeting_title, meeting_date, start_time, end_time, location, attendees, description) VALUES 
(1, '产品规划会议', '2024-12-25', '09:00:00', '11:00:00', '特斯拉会议室', '产品团队全体成员', '讨论2025年产品规划');

-- 13. 批量添加人员（示例）
INSERT INTO person_info (name, title, department, office_location, bio) VALUES 
('李四', '硬件工程师', '系统研发中心', '珠海总部15楼系统研发中心', '专注于硬件设计，有8年行业经验'),
('王五', '测试工程师', '机器人研发中心', '珠海总部14楼机器人研发中心', '负责产品测试和质量保证');

-- ========================================
-- 更新操作 (UPDATE)
-- ========================================

-- 14. 更新人员信息（按ID）
-- 注意：修改WHERE条件中的ID为实际要更新的人员ID
UPDATE person_info 
SET title = '高级软件工程师', 
    bio = '专注于芯片软件开发，有6年相关经验，现任技术小组组长'
WHERE id = 16;

-- 15. 更新人员办公地点
UPDATE person_info 
SET office_location = '珠海总部15楼IC研发中心新办公区'
WHERE department = 'IC研发中心';

-- 16. 更新会议信息
-- 注意：修改WHERE条件中的ID为实际要更新的会议ID
UPDATE meeting_schedule 
SET meeting_title = '产品战略规划会议',
    location = '爱因斯坦会议室',
    description = '讨论2025年产品战略规划和市场布局'
WHERE id = 1;

-- 17. 更新会议状态
UPDATE meeting_schedule 
SET status = 'completed'
WHERE meeting_date < DATE('now') AND status = 'scheduled';

-- 18. 延期会议
UPDATE meeting_schedule 
SET meeting_date = '2024-12-26',
    start_time = '14:00:00',
    end_time = '16:00:00'
WHERE id = 2;

-- ========================================
-- 删除操作 (DELETE)
-- ========================================

-- 19. 删除特定人员（注意：会级联删除相关会议）
-- 注意：修改WHERE条件为实际要删除的人员
-- DELETE FROM person_info WHERE name = '张三';

-- 20. 删除特定会议
-- 注意：修改WHERE条件中的ID为实际要删除的会议ID
-- DELETE FROM meeting_schedule WHERE id = 15;

-- 21. 删除已完成的会议
-- DELETE FROM meeting_schedule WHERE status = 'completed';

-- 22. 删除过期会议（7天前的已完成会议）
-- DELETE FROM meeting_schedule 
-- WHERE status = 'completed' 
-- AND meeting_date < DATE('now', '-7 days');

-- 23. 删除特定日期的会议
-- DELETE FROM meeting_schedule WHERE meeting_date = '2024-12-15';

-- ========================================
-- 高级查询示例
-- ========================================

-- 24. 查询最忙的人员（会议最多）
SELECT 
    p.name,
    p.title,
    COUNT(m.id) as meeting_count
FROM person_info p
LEFT JOIN meeting_schedule m ON p.id = m.person_id
GROUP BY p.id, p.name, p.title
ORDER BY meeting_count DESC
LIMIT 5;

-- 25. 查询会议室使用情况
SELECT 
    location,
    COUNT(*) as usage_count,
    MIN(meeting_date) as first_meeting,
    MAX(meeting_date) as last_meeting
FROM meeting_schedule
WHERE location IS NOT NULL AND location != ''
GROUP BY location
ORDER BY usage_count DESC;

-- 26. 查询即将到来的会议（未来7天）
SELECT 
    m.meeting_title,
    p.name as organizer,
    m.meeting_date,
    m.start_time,
    m.location
FROM meeting_schedule m
JOIN person_info p ON m.person_id = p.id
WHERE m.meeting_date BETWEEN DATE('now') AND DATE('now', '+7 days')
    AND m.status = 'scheduled'
ORDER BY m.meeting_date, m.start_time;

-- 27. 查询部门会议统计
SELECT 
    p.department,
    COUNT(m.id) as meeting_count,
    COUNT(DISTINCT m.meeting_date) as meeting_days
FROM person_info p
JOIN meeting_schedule m ON p.id = m.person_id
GROUP BY p.department
ORDER BY meeting_count DESC;

-- ========================================
-- 数据维护操作
-- ========================================

-- 28. 查看数据库统计信息
SELECT 
    (SELECT COUNT(*) FROM person_info) as total_persons,
    (SELECT COUNT(*) FROM meeting_schedule) as total_meetings,
    (SELECT COUNT(*) FROM meeting_schedule WHERE status = 'scheduled') as scheduled_meetings,
    (SELECT COUNT(*) FROM meeting_schedule WHERE meeting_date = DATE('now')) as today_meetings;

-- 29. 检查数据完整性
-- 查找没有会议的人员
SELECT * FROM person_info p
WHERE NOT EXISTS (SELECT 1 FROM meeting_schedule m WHERE m.person_id = p.id);

-- 查找孤儿会议（负责人不存在）
SELECT * FROM meeting_schedule m
WHERE NOT EXISTS (SELECT 1 FROM person_info p WHERE p.id = m.person_id);

-- 30. 清理测试数据（谨慎使用）
-- DELETE FROM meeting_schedule WHERE meeting_title LIKE '%测试%';
-- DELETE FROM person_info WHERE name LIKE '%测试%';
