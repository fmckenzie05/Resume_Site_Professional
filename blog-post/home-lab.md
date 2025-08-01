# Building My Home Lab: Network Segmentation & Monitoring

*Published on December 5, 2024 | Category: AUTOMATION*

## The Vision

As an IT professional transitioning from supply chain operations, I needed a comprehensive home lab to practice enterprise-level networking, security, and automation concepts. This lab needed to simulate real business environments while staying within a reasonable budget.

## Lab Architecture Overview

```
Internet ←→ [pfSense Firewall] ←→ [Managed Switch] ←→ [Lab Network]
                    ↓                      ↓
              [DMZ VLAN]            [Internal VLANs]
                    ↓                      ↓
            [Web Services]     [Domain Controller, Proxmox, etc.]
```

### Hardware Foundation
- **pfSense Box**: Protectli VP2420 (4-port Intel NICs)
- **Managed Switch**: UniFi Switch 24 PoE (24 ports + SFP+)
- **Hypervisor**: Dell OptiPlex 7070 (32GB RAM, 1TB NVMe)
- **NAS**: Synology DS218+ (2x4TB WD Red drives)
- **Wireless**: UniFi Access Point WiFi 6

**Total Investment**: ~$2,800 (spread over 6 months)

## Network Segmentation Strategy

### VLAN Design
```bash
# VLAN Configuration
VLAN 10: Management (192.168.10.0/24)    # Infrastructure access
VLAN 20: Lab Servers (192.168.20.0/24)   # VMs and containers
VLAN 30: DMZ (192.168.30.0/24)           # Public-facing services
VLAN 40: IoT (192.168.40.0/24)           # Smart home devices
VLAN 50: Guest (192.168.50.0/24)         # Visitor access
```

### Firewall Rules Implementation
```
# pfSense Rules Example
# Block inter-VLAN communication by default
pass out on LAN from LAN:network to !LAN:network keep state
block in on LAN from !LAN:network to LAN:network

# Allow specific services
pass in on IoT_VLAN proto tcp from IoT:network to LAB:192.168.20.100 port 53
pass in on GUEST_VLAN proto tcp from GUEST:network to any port { 80 443 53 }
```

## Virtual Infrastructure

### Proxmox VE Setup
Running multiple environments on a single hypervisor:

```bash
# VM Allocation
├── Domain Controller (Windows Server 2022) - 4GB RAM
├── Security Tools VM (Kali Linux) - 8GB RAM
├── SIEM Server (Ubuntu + ELK Stack) - 8GB RAM
├── Development Environment (Ubuntu) - 6GB RAM
└── Vulnerable VMs (Metasploitable, DVWA) - 4GB RAM
```

### Container Services
```yaml
# Docker Compose for monitoring stack
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
  
  node-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"
```

## Monitoring and Alerting

### ELK Stack Configuration
Centralized logging for security monitoring:

```json
{
  "logstash_pipeline": {
    "pfSense_logs": {
      "input": "syslog:514",
      "filter": "grok_pfSense_patterns",
      "output": "elasticsearch:9200"
    },
    "windows_events": {
      "input": "winlogbeat",
      "filter": "security_event_parsing",
      "output": "elasticsearch:9200"
    }
  }
}
```

### Key Dashboards Created
1. **Network Traffic Analysis**: Real-time bandwidth monitoring per VLAN
2. **Security Events**: Failed login attempts, blocked connections
3. **System Performance**: CPU, memory, disk usage across all VMs
4. **Vulnerability Tracking**: Nessus scan results integration

## Automation Projects

### PowerShell DSC for Windows Configuration
```powershell
Configuration LabDomainController {
    param(
        [string]$DomainName = "homelab.local",
        [string]$SafeModePassword
    )
    
    Import-DscResource -ModuleName PSDesiredStateConfiguration
    Import-DscResource -ModuleName ActiveDirectoryDsc
    
    Node $AllNodes.NodeName {
        WindowsFeature ADDSInstall {
            Ensure = "Present"
            Name = "AD-Domain-Services"
        }
        
        ADDomain FirstDS {
            DomainName = $DomainName
            Credential = $DomainAdministratorCredential
            SafemodeAdministratorPassword = $SafeModeAdministratorCredential
            DependsOn = "[WindowsFeature]ADDSInstall"
        }
    }
}
```

### Ansible Playbooks for Linux Systems
```yaml
# playbook.yml - Security hardening
---
- hosts: lab_servers
  become: yes
  tasks:
    - name: Update all packages
      apt:
        upgrade: dist
        update_cache: yes
    
    - name: Configure UFW firewall
      ufw:
        rule: "{{ item.rule }}"
        port: "{{ item.port }}"
        proto: "{{ item.proto }}"
      loop:
        - { rule: 'allow', port: '22', proto: 'tcp' }
        - { rule: 'allow', port: '80', proto: 'tcp' }
        - { rule: 'allow', port: '443', proto: 'tcp' }
    
    - name: Enable fail2ban
      systemd:
        name: fail2ban
        enabled: yes
        state: started
```

## Security Testing Environment

### Vulnerable Applications Setup
Created isolated environment for ethical hacking practice:

```docker
# docker-compose.yml for vulnerable apps
version: '3'
services:
  dvwa:
    image: vulnerables/web-dvwa
    ports:
      - "8080:80"
    networks:
      - vulnerable_net
  
  webgoat:
    image: webgoat/webgoat-8.0
    ports:
      - "8081:8080"
    networks:
      - vulnerable_net

networks:
  vulnerable_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Penetration Testing Workflow
1. **Reconnaissance**: Nmap network discovery
2. **Vulnerability Assessment**: OpenVAS automated scans
3. **Exploitation**: Metasploit framework testing
4. **Post-Exploitation**: Privilege escalation practice
5. **Reporting**: Automated report generation

## Real-World Learning Outcomes

### Skills Developed
- **Network Architecture**: VLAN design, routing, firewalling
- **Virtualization**: Proxmox management, resource allocation
- **Monitoring**: ELK stack, Grafana dashboards, alerting
- **Automation**: Ansible, PowerShell DSC, Docker orchestration
- **Security**: Vulnerability management, incident response

### Certification Preparation
This lab directly supported my preparation for:
- **CompTIA Security+** ✅ (Passed November 2024)
- **AWS Solutions Architect** (Scheduled February 2025)
- **Cisco CCNA** (Planning for June 2025)

## Cost-Benefit Analysis

### Initial Investment Breakdown
- Hardware: $2,200
- Software Licenses: $400 (Windows Server, VMware initially)
- Power/Internet: $40/month additional

### Value Gained
- **Hands-on Experience**: Enterprise-level networking and security
- **Certification Prep**: $3,000+ in training course value
- **Career Development**: Practical skills for IT operations roles
- **Home Network**: Enhanced security and performance

**ROI**: Lab paid for itself within 6 months through improved job prospects and salary negotiations.

## Future Enhancements

### Planned Upgrades
1. **Kubernetes Cluster**: 3-node setup for container orchestration
2. **Cloud Integration**: Site-to-site VPN with AWS
3. **Advanced Threat Detection**: SOAR platform implementation
4. **Zero Trust Architecture**: Implement concepts from work projects

### Learning Roadmap
- **Red Team Tactics**: Advanced penetration testing
- **Cloud Security**: AWS security services deep dive
- **DevSecOps**: CI/CD pipeline security integration
- **Compliance**: SOC 2, PCI-DSS framework implementation

## Lessons Learned

### What Worked Well
1. **Start Simple**: Basic pfSense setup before advanced features
2. **Document Everything**: Wiki with all configurations and procedures
3. **Regular Backups**: VM snapshots before major changes
4. **Community Resources**: r/homelab and Discord communities invaluable

### Common Pitfalls Avoided
1. **Over-Engineering**: Resisted urge to build everything at once
2. **Budget Creep**: Stuck to planned budget with phased approach
3. **Single Points of Failure**: Redundancy in critical services
4. **Security Neglect**: Treated lab with production-level security

## Resources and References

### Essential Tools
- **pfSense**: Open-source firewall/router platform
- **Proxmox VE**: Enterprise virtualization platform
- **UniFi Controller**: Network device management
- **ELK Stack**: Elasticsearch, Logstash, Kibana for logging
- **Grafana**: Metrics visualization and alerting

### Learning Resources
- [pfSense Documentation](https://docs.netgate.com/pfsense)
- [Proxmox VE Wiki](https://pve.proxmox.com/wiki)
- [Homelab Subreddit](https://reddit.com/r/homelab)
- [TechnoTim YouTube Channel](https://youtube.com/c/TechnoTimLive)

### Scripts and Configurations
All configuration files, scripts, and documentation available in my [GitHub repository](https://github.com/fmckenzie05/homelab-infrastructure).

---

**Thinking about building your own lab?** Connect with me on [LinkedIn](https://www.linkedin.com/in/fernandomckenzie/) - I'm happy to share lessons learned and help you avoid common pitfalls.

*Next post: "Implementing Zero-Trust Architecture in Legacy Supply Chain Systems" - Coming December 15th* 