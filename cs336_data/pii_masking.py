"""
PII (Personally Identifiable Information) Masking Module.

This module provides functions to detect and mask sensitive personal information
including email addresses, phone numbers, and IP addresses.
"""

import re
from typing import Tuple


# Mask replacement strings
EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"


def mask_emails(text: str) -> Tuple[str, int]:
    """
    Mask email addresses in the given text.
    
    Replaces email addresses with |||EMAIL_ADDRESS|||.
    Does not count already-masked emails (existing |||EMAIL_ADDRESS||| strings).
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (masked_text, num_masked) where num_masked is the count
        of email addresses that were masked in this call.
    """
    if not text or not isinstance(text, str):
        return (text or "", 0)
    
    # Email regex pattern
    # Matches: user@domain.com, user.name@domain.co.uk, user+tag@domain.org, etc.
    email_pattern = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )
    
    # Find all matches
    matches = email_pattern.findall(text)
    num_masked = len(matches)
    
    # Replace all email addresses with mask
    masked_text = email_pattern.sub(EMAIL_MASK, text)
    
    return (masked_text, num_masked)


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """
    Mask phone numbers in the given text.
    
    Replaces phone numbers with |||PHONE_NUMBER|||.
    Handles various formats:
    - 2831823829 (10 digits)
    - (283)-182-3829
    - (283) 182 3829
    - 283-182-3829
    - 283.182.3829
    - +1 283 182 3829
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (masked_text, num_masked) where num_masked is the count
        of phone numbers that were masked.
    """
    if not text or not isinstance(text, str):
        return (text or "", 0)
  
    # Phone number patterns - order matters, more specific patterns first
    phone_patterns = [
        # International format with country code: +1 (283) 182-3829 or +1-283-182-3829
        r'\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        # Format: (283)-182-3829 or (283) 182-3829 or (283)182-3829
        r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}\b',
        # Format: 283-182-3829 or 283.182.3829 or 283 182 3829
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
        # Format: 2831823829 (10 consecutive digits)
        r'\b\d{10}\b',
    ]
    
    # Combine patterns
    combined_pattern = re.compile('|'.join(phone_patterns))
    
    # Find all matches
    matches = combined_pattern.findall(text)
    num_masked = len(matches)
    
    # Replace all phone numbers with mask
    masked_text = combined_pattern.sub(PHONE_MASK, text)
    
    return (masked_text, num_masked)


def mask_ip_addresses(text: str) -> Tuple[str, int]:
    """
    Mask IP addresses in the given text.
    
    Replaces IPv4 addresses with |||IP_ADDRESS|||.
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (masked_text, num_masked) where num_masked is the count
        of IP addresses that were masked.
    """
    if not text or not isinstance(text, str):
        return (text or "", 0)
    
    # IPv4 pattern: matches 0.0.0.0 to 255.255.255.255
    # Each octet: 0-255
    octet = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    ipv4_pattern = re.compile(
        rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'
    )
    
    # Find all matches
    matches = ipv4_pattern.findall(text)
    num_masked = len(matches)
    
    # Replace all IP addresses with mask
    masked_text = ipv4_pattern.sub(IP_MASK, text)
    
    return (masked_text, num_masked)


def mask_all_pii(text: str) -> Tuple[str, dict]:
    """
    Mask all PII (emails, phone numbers, IP addresses) in the given text.
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (masked_text, stats_dict) where stats_dict contains
        counts for each type of PII masked.
    """
    if not text or not isinstance(text, str):
        return (text or "", {"emails": 0, "phones": 0, "ips": 0, "total": 0})
    
    stats = {}
    
    # Mask emails first
    text, stats["emails"] = mask_emails(text)
    
    # Mask phone numbers
    text, stats["phones"] = mask_phone_numbers(text)
    
    # Mask IP addresses
    text, stats["ips"] = mask_ip_addresses(text)
    
    stats["total"] = stats["emails"] + stats["phones"] + stats["ips"]
    
    return (text, stats)

