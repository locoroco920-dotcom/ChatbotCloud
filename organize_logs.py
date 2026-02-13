#!/usr/bin/env python
"""
Log Organizer - Groups all Q&A pairs by conversation session_id
Reads the raw questions_log.txt and reorganizes it so each conversation
is kept together in full, even if multiple conversations were happening at the same time.
"""

import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'questions_log.txt')
ORGANIZED_LOG = os.path.join(SCRIPT_DIR, 'questions_log_organized.txt')

def organize_logs():
    """Read log file and reorganize by session_id"""
    
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] Log file not found: {LOG_FILE}")
        return
    
    # Read the entire log file
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by conversation headers
    conversation_blocks = re.split(r'={60}\nCONVERSATION:', content)
    
    conversations = defaultdict(lambda: {'started': '', 'qa_pairs': []})
    
    for block in conversation_blocks:
        lines = block.split('\n')
        if not lines:
            continue
        
        # First line should be the session_id
        session_line = lines[0].strip()
        match = re.search(r'([a-f0-9\-]+)', session_line)
        if not match:
            continue
        
        session_id = match.group(1)
        
        # Find Started time
        started = ""
        for line in lines:
            if 'Started:' in line:
                started = line.split('Started:')[1].strip()
                break
        
        conversations[session_id]['started'] = started
        
        # Extract all Q&A pairs
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.strip().startswith('Q:'):
                q = line.strip()[2:].strip()
                
                # Find corresponding A:
                i += 1
                a_lines = []
                while i < len(lines):
                    next_line = lines[i]
                    
                    # Stop at next Q or end of block
                    if next_line.strip().startswith('Q:'):
                        i -= 1
                        break
                    
                    if next_line.strip().startswith('A:'):
                        a_lines.append(next_line.strip()[2:].strip())
                    elif a_lines:  # We've started capturing the answer
                        # Stop if we hit a blank line followed by Q (new pair)
                        if not next_line.strip():
                            j = i + 1
                            if j < len(lines) and lines[j].strip().startswith('Q:'):
                                break
                        if next_line.strip():
                            a_lines.append(next_line)
                    
                    i += 1
                
                a = '\n'.join(a_lines).strip()
                if q and a:
                    conversations[session_id]['qa_pairs'].append((q, a))
            
            i += 1
    
    # Write organized log
    with open(ORGANIZED_LOG, 'w', encoding='utf-8') as f:
        for session_id in sorted(conversations.keys()):
            data = conversations[session_id]
            f.write(f"\n{'='*60}\n")
            f.write(f"CONVERSATION: {session_id}\n")
            f.write(f"Started: {data['started']}\n")
            f.write(f"{'='*60}\n\n")
            
            for q, a in data['qa_pairs']:
                f.write(f"Q: {q}\n")
                f.write(f"A: {a}\n\n")
    
    print(f"[OK] Organized log created: {ORGANIZED_LOG}")
    print(f"[OK] Total conversations: {len(conversations)}")
    
    # Show summary
    for session_id in sorted(conversations.keys()):
        num_qa = len(conversations[session_id]['qa_pairs'])
        print(f"     {session_id[:12]}... : {num_qa} Q&A pairs")

if __name__ == "__main__":
    print("Organizing chat logs...\n")
    organize_logs()
    print("\nDone! Check questions_log_organized.txt for the grouped conversations.")

