import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



# ─────────────────────────────────────────────
# 1. NIST NVD — Latest CVEs
# ─────────────────────────────────────────────
def fetch_nist_cves():
    print("📡 Fetching latest CVEs from NIST NVD...")
    docs = []
    try:
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=200"
        res = requests.get(url, timeout=30)
        data = res.json()
        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "Unknown")
            descriptions = cve.get("descriptions", [])
            desc = next((d["value"] for d in descriptions if d["lang"] == "en"), "No description")
            published = cve.get("published", "")[:10]
            severity = "Unknown"
            try:
                metrics = cve.get("metrics", {})
                cvss = metrics.get("cvssMetricV31", metrics.get("cvssMetricV30", metrics.get("cvssMetricV2", [])))
                if cvss:
                    severity = cvss[0]["cvssData"].get("baseSeverity", "Unknown")
            except Exception:
                pass
            text = (
                f"CVE ID: {cve_id}\n"
                f"Published: {published}\n"
                f"Severity: {severity}\n"
                f"Description: {desc}"
            )
            docs.append(Document(page_content=text, metadata={"source": "NIST NVD", "cve_id": cve_id}))
        print(f"✅ Fetched {len(docs)} CVEs from NIST NVD")
    except Exception as e:
        print(f"⚠️ NIST NVD fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 2. MITRE ATT&CK — Enterprise Framework
# ─────────────────────────────────────────────
def fetch_mitre_attack():
    print("📡 Fetching MITRE ATT&CK Enterprise...")
    docs = []
    try:
        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        res = requests.get(url, timeout=90)
        data = res.json()
        for obj in data.get("objects", []):
            obj_type = obj.get("type", "")
            name = obj.get("name", "Unknown")
            description = obj.get("description", "")
            if not description or len(description) < 30:
                continue
            if obj_type == "attack-pattern":
                tactics = [p.get("phase_name", "").replace("-", " ").title() for p in obj.get("kill_chain_phases", [])]
                platforms = ", ".join(obj.get("x_mitre_platforms", []))
                text = (
                    f"MITRE ATT&CK Technique: {name}\n"
                    f"Tactics: {', '.join(tactics)}\n"
                    f"Platforms: {platforms}\n"
                    f"Description: {description[:2000]}"
                )
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "technique"}))
            elif obj_type == "intrusion-set":
                aliases = ", ".join(obj.get("aliases", []))
                text = (
                    f"Threat Actor Group: {name}\n"
                    f"Aliases: {aliases}\n"
                    f"Description: {description[:2000]}"
                )
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "threat_actor"}))
            elif obj_type == "malware":
                platforms = ", ".join(obj.get("x_mitre_platforms", []))
                text = (
                    f"Malware: {name}\n"
                    f"Platforms: {platforms}\n"
                    f"Description: {description[:2000]}"
                )
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "malware"}))
            elif obj_type == "tool":
                text = f"Security Tool: {name}\nDescription: {description[:2000]}"
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "tool"}))
            elif obj_type == "course-of-action":
                text = f"Mitigation: {name}\nDescription: {description[:2000]}"
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "mitigation"}))
            elif obj_type == "campaign":
                text = f"Campaign: {name}\nDescription: {description[:2000]}"
                docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK", "type": "campaign"}))
        print(f"✅ Fetched {len(docs)} entries from MITRE ATT&CK Enterprise")
    except Exception as e:
        print(f"⚠️ MITRE ATT&CK Enterprise fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 3. MITRE ATT&CK — Mobile Framework
# ─────────────────────────────────────────────
def fetch_mitre_mobile():
    print("📡 Fetching MITRE ATT&CK Mobile...")
    docs = []
    try:
        url = "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
        res = requests.get(url, timeout=90)
        data = res.json()
        for obj in data.get("objects", []):
            name = obj.get("name", "Unknown")
            description = obj.get("description", "")
            if obj.get("type") != "attack-pattern" or not description or len(description) < 30:
                continue
            tactics = [p.get("phase_name", "").replace("-", " ").title() for p in obj.get("kill_chain_phases", [])]
            text = (
                f"MITRE Mobile ATT&CK Technique: {name}\n"
                f"Tactics: {', '.join(tactics)}\n"
                f"Description: {description[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK Mobile", "type": "technique"}))
        print(f"✅ Fetched {len(docs)} entries from MITRE ATT&CK Mobile")
    except Exception as e:
        print(f"⚠️ MITRE ATT&CK Mobile fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 4. MITRE ATT&CK — ICS Framework
# ─────────────────────────────────────────────
def fetch_mitre_ics():
    print("📡 Fetching MITRE ATT&CK ICS...")
    docs = []
    try:
        url = "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"
        res = requests.get(url, timeout=90)
        data = res.json()
        for obj in data.get("objects", []):
            name = obj.get("name", "Unknown")
            description = obj.get("description", "")
            if obj.get("type") != "attack-pattern" or not description or len(description) < 30:
                continue
            tactics = [p.get("phase_name", "").replace("-", " ").title() for p in obj.get("kill_chain_phases", [])]
            text = (
                f"MITRE ICS ATT&CK Technique: {name}\n"
                f"Tactics: {', '.join(tactics)}\n"
                f"Description: {description[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "MITRE ATT&CK ICS", "type": "technique"}))
        print(f"✅ Fetched {len(docs)} entries from MITRE ATT&CK ICS")
    except Exception as e:
        print(f"⚠️ MITRE ATT&CK ICS fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 5. MITRE D3FEND — Defensive Techniques
# ─────────────────────────────────────────────
def fetch_mitre_d3fend():
    print("📡 Fetching MITRE D3FEND defensive techniques...")
    docs = []
    try:
        url = "https://d3fend.mitre.org/ontologies/d3fend.json"
        res = requests.get(url, timeout=30)
        data = res.json()
        for technique in data.get("techniques", []):
            name = technique.get("label", {}).get("value", "Unknown")
            definition = technique.get("definition", {}).get("value", "")
            if not definition or len(definition) < 30:
                continue
            text = (
                f"MITRE D3FEND Defensive Technique: {name}\n"
                f"Definition: {definition[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "MITRE D3FEND", "type": "defensive_technique"}))
        print(f"✅ Fetched {len(docs)} entries from MITRE D3FEND")
    except Exception as e:
        print(f"⚠️ MITRE D3FEND fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 6. CISA — Known Exploited Vulnerabilities (KEV)
# ─────────────────────────────────────────────
def fetch_cisa_kev():
    print("📡 Fetching CISA Known Exploited Vulnerabilities...")
    docs = []
    try:
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        res = requests.get(url, timeout=20)
        data = res.json()
        for vuln in data.get("vulnerabilities", []):
            text = (
                f"CISA Known Exploited Vulnerability\n"
                f"CVE: {vuln.get('cveID', 'Unknown')}\n"
                f"Vendor: {vuln.get('vendorProject', '')}\n"
                f"Product: {vuln.get('product', '')}\n"
                f"Vulnerability: {vuln.get('vulnerabilityName', '')}\n"
                f"Description: {vuln.get('shortDescription', '')}\n"
                f"Required Action: {vuln.get('requiredAction', '')}\n"
                f"Due Date: {vuln.get('dueDate', '')}"
            )
            docs.append(Document(page_content=text, metadata={"source": "CISA KEV", "cve_id": vuln.get("cveID", "")}))
        print(f"✅ Fetched {len(docs)} CISA KEV entries")
    except Exception as e:
        print(f"⚠️ CISA KEV fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 7. CISA — Security Advisories RSS
# ─────────────────────────────────────────────
def fetch_cisa_advisories():
    print("📡 Fetching CISA security advisories...")
    docs = []
    try:
        url = "https://www.cisa.gov/cybersecurity-advisories/all.xml"
        res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:60]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            link = item.find("link").get_text(strip=True) if item.find("link") else ""
            if not description or len(description) < 30:
                continue
            text = (
                f"CISA Advisory: {title}\n"
                f"Published: {pub_date}\n"
                f"Link: {link}\n"
                f"Summary: {description[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "CISA Advisory"}))
        print(f"✅ Fetched {len(docs)} CISA advisories")
    except Exception as e:
        print(f"⚠️ CISA advisories fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 8. US-CERT Alerts RSS
# ─────────────────────────────────────────────
def fetch_uscert_alerts():
    print("📡 Fetching US-CERT alerts...")
    docs = []
    try:
        url = "https://www.cisa.gov/uscert/ncas/alerts.xml"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:40]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            if not description or len(description) < 50:
                continue
            clean_desc = BeautifulSoup(description, "html.parser").get_text()
            text = (
                f"US-CERT Alert: {title}\n"
                f"Published: {pub_date}\n"
                f"Details: {clean_desc[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "US-CERT"}))
        print(f"✅ Fetched {len(docs)} US-CERT alerts")
    except Exception as e:
        print(f"⚠️ US-CERT alerts fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 9. OWASP — Web Top 10
# ─────────────────────────────────────────────
def fetch_owasp_top10():
    print("📡 Fetching OWASP Top 10...")
    docs = []
    try:
        url = "https://owasp.org/www-project-top-ten/"
        res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
        if text:
            docs.append(Document(page_content=text, metadata={"source": "OWASP Top 10"}))
            print("✅ Fetched OWASP Top 10")
    except Exception as e:
        print(f"⚠️ OWASP Top 10 fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 10. OWASP — API Security Top 10
# ─────────────────────────────────────────────
def fetch_owasp_api():
    print("📡 Fetching OWASP API Security Top 10...")
    docs = []
    try:
        url = "https://owasp.org/www-project-api-security/"
        res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
        if text:
            docs.append(Document(page_content=text, metadata={"source": "OWASP API Security Top 10"}))
            print("✅ Fetched OWASP API Security Top 10")
    except Exception as e:
        print(f"⚠️ OWASP API Security fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 11. OWASP — Mobile Security Top 10
# ─────────────────────────────────────────────
def fetch_owasp_mobile():
    print("📡 Fetching OWASP Mobile Security Top 10...")
    docs = []
    try:
        url = "https://owasp.org/www-project-mobile-top-10/"
        res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
        if text:
            docs.append(Document(page_content=text, metadata={"source": "OWASP Mobile Top 10"}))
            print("✅ Fetched OWASP Mobile Top 10")
    except Exception as e:
        print(f"⚠️ OWASP Mobile fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 12. OWASP — Testing Guide
# ─────────────────────────────────────────────
def fetch_owasp_testing_guide():
    print("📡 Fetching OWASP Testing Guide...")
    docs = []
    pages = [
        "https://owasp.org/www-project-web-security-testing-guide/",
        "https://owasp.org/www-project-web-security-testing-guide/v42/4-Web_Application_Security_Testing/",
    ]
    for url in pages:
        try:
            res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
            if text:
                docs.append(Document(page_content=text[:5000], metadata={"source": "OWASP Testing Guide", "url": url}))
        except Exception as e:
            print(f"⚠️ OWASP Testing Guide fetch failed for {url}: {e}")
    print(f"✅ Fetched {len(docs)} OWASP Testing Guide pages")
    return docs


# ─────────────────────────────────────────────
# 13. Cyber Kill Chain — Live Sources
# ─────────────────────────────────────────────
def fetch_cyber_kill_chain():
    print("📡 Fetching Cyber Kill Chain...")
    docs = []
    urls = [
        "https://www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-chain.html",
        "https://en.wikipedia.org/wiki/Kill_chain_(military)",
        "https://en.wikipedia.org/wiki/Cyber_kill_chain",
    ]
    for url in urls:
        try:
            res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
            if text:
                docs.append(Document(page_content=text[:5000], metadata={"source": "Cyber Kill Chain", "url": url}))
        except Exception as e:
            print(f"⚠️ Cyber Kill Chain fetch failed for {url}: {e}")
    print(f"✅ Fetched {len(docs)} Cyber Kill Chain pages")
    return docs


# ─────────────────────────────────────────────
# 14. HackTricks — Pentesting Knowledge
# ─────────────────────────────────────────────
def fetch_hacktricks():
    print("📡 Fetching HackTricks content...")
    docs = []
    pages = [
        "https://book.hacktricks.xyz/generic-methodologies-and-resources/external-recon-methodology",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-web",
        "https://book.hacktricks.xyz/generic-methodologies-and-resources/shells/linux",
        "https://book.hacktricks.xyz/windows-hardening/windows-local-privilege-escalation",
        "https://book.hacktricks.xyz/generic-methodologies-and-resources/reverse-shells/linux",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-smb",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-ftp",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-ssh",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-dns",
        "https://book.hacktricks.xyz/network-services-pentesting/pentesting-rdp",
        "https://book.hacktricks.xyz/linux-hardening/privilege-escalation",
        "https://book.hacktricks.xyz/pentesting-web/sql-injection",
        "https://book.hacktricks.xyz/pentesting-web/xss-cross-site-scripting",
        "https://book.hacktricks.xyz/pentesting-web/ssrf-server-side-request-forgery",
        "https://book.hacktricks.xyz/pentesting-web/file-inclusion",
        "https://book.hacktricks.xyz/pentesting-web/xxe-xee-xml-external-entity",
        "https://book.hacktricks.xyz/pentesting-web/deserialization",
        "https://book.hacktricks.xyz/pentesting-web/csrf-cross-site-request-forgery",
        "https://book.hacktricks.xyz/generic-methodologies-and-resources/brute-force",
        "https://book.hacktricks.xyz/generic-methodologies-and-resources/python/bypass-python-sandboxes",
    ]
    for url in pages:
        try:
            res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            text = "\n".join(p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50)
            if text:
                docs.append(Document(page_content=text[:5000], metadata={"source": "HackTricks", "url": url}))
        except Exception as e:
            print(f"⚠️ HackTricks fetch failed for {url}: {e}")
    print(f"✅ Fetched {len(docs)} HackTricks pages")
    return docs


# ─────────────────────────────────────────────
# 15. Exploit Database — Public Exploits RSS
# ─────────────────────────────────────────────
def fetch_exploit_db():
    print("📡 Fetching Exploit-DB recent entries...")
    docs = []
    try:
        url = "https://www.exploit-db.com/rss.xml"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:50]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            link = item.find("link").get_text(strip=True) if item.find("link") else ""
            if not title:
                continue
            text = (
                f"Exploit-DB Entry: {title}\n"
                f"Link: {link}\n"
                f"Details: {description[:1500]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "Exploit-DB"}))
        print(f"✅ Fetched {len(docs)} Exploit-DB entries")
    except Exception as e:
        print(f"⚠️ Exploit-DB fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 16. SANS Internet Storm Center
# ─────────────────────────────────────────────
def fetch_sans_isc():
    print("📡 Fetching SANS Internet Storm Center diaries...")
    docs = []
    try:
        url = "https://isc.sans.edu/rssfeed_full.xml"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:40]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            if not description or len(description) < 50:
                continue
            clean_desc = BeautifulSoup(description, "html.parser").get_text()
            text = (
                f"SANS ISC Diary: {title}\n"
                f"Published: {pub_date}\n"
                f"Content: {clean_desc[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "SANS ISC"}))
        print(f"✅ Fetched {len(docs)} SANS ISC diaries")
    except Exception as e:
        print(f"⚠️ SANS ISC fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 17. Krebs on Security
# ─────────────────────────────────────────────
def fetch_krebs_on_security():
    print("📡 Fetching Krebs on Security articles...")
    docs = []
    try:
        url = "https://krebsonsecurity.com/feed/"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:30]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            if not description or len(description) < 50:
                continue
            clean_desc = BeautifulSoup(description, "html.parser").get_text()
            text = (
                f"Krebs on Security: {title}\n"
                f"Published: {pub_date}\n"
                f"Summary: {clean_desc[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "Krebs on Security"}))
        print(f"✅ Fetched {len(docs)} Krebs on Security articles")
    except Exception as e:
        print(f"⚠️ Krebs on Security fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 18. The Hacker News
# ─────────────────────────────────────────────
def fetch_hacker_news():
    print("📡 Fetching The Hacker News articles...")
    docs = []
    try:
        url = "https://feeds.feedburner.com/TheHackersNews"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for item in soup.find_all("item")[:40]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            description = item.find("description").get_text(strip=True) if item.find("description") else ""
            pub_date = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            if not description or len(description) < 50:
                continue
            clean_desc = BeautifulSoup(description, "html.parser").get_text()
            text = (
                f"The Hacker News: {title}\n"
                f"Published: {pub_date}\n"
                f"Summary: {clean_desc[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "The Hacker News"}))
        print(f"✅ Fetched {len(docs)} The Hacker News articles")
    except Exception as e:
        print(f"⚠️ The Hacker News fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 19. Schneier on Security
# ─────────────────────────────────────────────
def fetch_schneier():
    print("📡 Fetching Schneier on Security articles...")
    docs = []
    try:
        url = "https://www.schneier.com/feed/atom/"
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.content, "xml")
        for entry in soup.find_all("entry")[:30]:
            title = entry.find("title").get_text(strip=True) if entry.find("title") else ""
            content = entry.find("content")
            summary = entry.find("summary")
            raw = content or summary
            if not raw:
                continue
            clean = BeautifulSoup(raw.get_text(), "html.parser").get_text()
            if len(clean) < 50:
                continue
            text = (
                f"Schneier on Security: {title}\n"
                f"Content: {clean[:2000]}"
            )
            docs.append(Document(page_content=text, metadata={"source": "Schneier on Security"}))
        print(f"✅ Fetched {len(docs)} Schneier on Security articles")
    except Exception as e:
        print(f"⚠️ Schneier on Security fetch failed: {e}")
    return docs


# ─────────────────────────────────────────────
# 20. CVE Details — Vulnerability Stats
# ─────────────────────────────────────────────
def fetch_cve_details():
    print("📡 Fetching CVE Details stats...")
    docs = []
    urls = [
        "https://www.cvedetails.com/vulnerabilities-by-types.php",
        "https://www.cvedetails.com/top-50-products.php",
    ]
    for url in urls:
        try:
            res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            text = "\n".join(el.get_text() for el in soup.find_all(["p", "td", "th"]) if len(el.get_text()) > 30)
            if text:
                docs.append(Document(page_content=text[:5000], metadata={"source": "CVE Details", "url": url}))
        except Exception as e:
            print(f"⚠️ CVE Details fetch failed for {url}: {e}")
    print(f"✅ Fetched {len(docs)} CVE Details pages")
    return docs


# ─────────────────────────────────────────────
# 21. Wikipedia — Core Cybersecurity Concepts
# ─────────────────────────────────────────────
def fetch_wikipedia_concepts():
    print("📡 Fetching cybersecurity concepts from Wikipedia...")
    docs = []
    topics = [
        "Cyber_kill_chain",
        "Penetration_test",
        "Vulnerability_(computing)",
        "Exploit_(computer_security)",
        "Malware",
        "Ransomware",
        "Phishing",
        "Social_engineering_(security)",
        "Man-in-the-middle_attack",
        "SQL_injection",
        "Cross-site_scripting",
        "Denial-of-service_attack",
        "Zero-day_(computing)",
        "Advanced_persistent_threat",
        "Intrusion_detection_system",
        "Firewall_(computing)",
        "Public_key_infrastructure",
        "Multi-factor_authentication",
        "Security_information_and_event_management",
        "Threat_intelligence",
        "Incident_management",
        "Digital_forensics",
        "Cryptography",
        "Symmetric-key_algorithm",
        "Public-key_cryptography",
        "Transport_Layer_Security",
        "Virtual_private_network",
        "Endpoint_security",
        "Patch_(computing)",
        "Privilege_escalation",
        "Lateral_movement_(cybersecurity)",
        "Command_and_control_(malware)",
        "Botnet",
        "Rootkit",
        "Keylogger",
        "Spyware",
        "Trojan_horse_(computing)",
        "Cyber_espionage",
        "Bug_bounty_program",
        "Red_team",
        "Blue_team_(computer_security)",
        "Threat_hunting",
        "Security_operations_center",
        "MITRE_ATT%26CK",
        "Common_Vulnerability_Scoring_System",
        "Common_Vulnerabilities_and_Exposures",
        "Network_security",
        "Cloud_computing_security",
        "DevSecOps",
        "Identity_and_access_management",
        "Zero_trust_security_model",
        "Password_cracking",
        "Brute-force_attack",
        "Dictionary_attack",
        "Rainbow_table",
        "ARP_spoofing",
        "DNS_spoofing",
        "Session_hijacking",
        "Clickjacking",
        "Steganography",
        "Honeypot_(computing)",
        "Cyber_threat_intelligence",
        "Security_awareness",
        "Vulnerability_management",
    ]
    for topic in topics:
        try:
            url = f"https://en.wikipedia.org/wiki/{topic}"
            res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                continue
            text = "\n".join(p.get_text() for p in content_div.find_all("p") if len(p.get_text()) > 50)
            if text:
                docs.append(Document(
                    page_content=text[:5000],
                    metadata={"source": "Wikipedia", "topic": topic.replace("_", " ")}
                ))
        except Exception as e:
            print(f"⚠️ Wikipedia fetch failed for {topic}: {e}")
    print(f"✅ Fetched {len(docs)} Wikipedia cybersecurity articles")
    return docs


# ─────────────────────────────────────────────
# 22. Local Knowledge Base
# ─────────────────────────────────────────────
def load_local_knowledge():
    print("📂 Loading local knowledge base...")
    docs = []
    try:
        with open("knowledge/cybersecurity.txt", "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": "Local Knowledge Base"}))
            print("✅ Loaded local knowledge base")
    except Exception as e:
        print(f"⚠️ Local knowledge load failed: {e}")
    return docs


# ─────────────────────────────────────────────
# MAIN — Build Vector Store
# ─────────────────────────────────────────────
def create_vectorstore():
    all_docs = []

    all_docs.extend(fetch_nist_cves())
    all_docs.extend(fetch_mitre_attack())
    all_docs.extend(fetch_mitre_mobile())
    all_docs.extend(fetch_mitre_ics())
    all_docs.extend(fetch_mitre_d3fend())
    all_docs.extend(fetch_cisa_kev())
    all_docs.extend(fetch_cisa_advisories())
    all_docs.extend(fetch_uscert_alerts())
    all_docs.extend(fetch_owasp_top10())
    all_docs.extend(fetch_owasp_api())
    all_docs.extend(fetch_owasp_mobile())
    all_docs.extend(fetch_owasp_testing_guide())
    all_docs.extend(fetch_cyber_kill_chain())
    all_docs.extend(fetch_hacktricks())
    all_docs.extend(fetch_exploit_db())
    all_docs.extend(fetch_sans_isc())
    all_docs.extend(fetch_krebs_on_security())
    all_docs.extend(fetch_hacker_news())
    all_docs.extend(fetch_schneier())
    all_docs.extend(fetch_cve_details())
    all_docs.extend(fetch_wikipedia_concepts())
    all_docs.extend(load_local_knowledge())

    if not all_docs:
        print("❌ No documents fetched. Aborting.")
        return

    print(f"\n📚 Total documents collected: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks")

    print("🔢 Creating embeddings — this may take several minutes...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")

    print("\n✅ Knowledge base created successfully!")
    print("📊 Sources included:")
    sources = {}
    for doc in all_docs:
        src = doc.metadata.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"   • {src}: {count} documents")


if __name__ == "__main__":
    create_vectorstore()