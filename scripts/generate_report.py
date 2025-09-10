import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import os 
from datetime import datetime 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import json 
import argparse 

def load_text_file(Path):
    with open(Path, encoding="utf-8") as jsonfile:
        return json.load(jsonfile)

def date_to_datetime_grouped(dataset: pd.DataFrame) -> pd.DataFrame:
    ''' This function is the first one called for every csv that we want to treat, 
    it sets all dates to datetime and create a new column 'date' for futur statistics day by day'''
    
    dataset['started_at'] = pd.to_datetime(dataset['started_at'])
    dataset['last_seen_at'] = pd.to_datetime(dataset['last_seen_at'])

    dataset['date'] = dataset['started_at'].dt.date
    dataset['hour'] = dataset['started_at'].dt.hour
    dataset["site"] = dataset["name"].str.replace(r"-\d+$","", regex=True)

    return dataset 

def days_and_cameras(dataset):
    dmin = dataset['date'].min()
    dmax = dataset['date'].max()
    nb_days = (dmax - dmin).days + 1 
    nb_cameras = dataset['name'].nunique()
    nb_sites = dataset['site'].unique()
    return nb_days, nb_cameras, nb_sites 

def first_statistics(dataset: pd.DataFrame):
    '''This function is returning a dictionnary with all the statistics we want from the csv
    for futur plotting'''
    detections_per_day = dataset.groupby('date')['id'].count()
    annoted_fires = dataset[dataset['is_wildfire'].notna()]['id'].count()
    annoted_per_day = dataset[dataset['is_wildfire'].notna()].groupby('date')['id'].count()
    unlabeled = dataset[dataset['is_wildfire'].isna()]['id'].count()
    false_positives = dataset[dataset['is_wildfire'] == False]['id'].count()
    false_positives_per_day = dataset[dataset['is_wildfire'] == False].groupby('date')['id'].count()
    true_positives = dataset[dataset['is_wildfire'] == True]['id'].count()
    true_positives_per_day = dataset[dataset['is_wildfire'] == True].groupby('date')['id'].count()
    detections_per_hour = dataset.groupby('hour')['id'].count()
    detections_per_hour_site = dataset.groupby(['site', 'hour'])['id'].count().unstack(fill_value=0)

    stats = {
        "annoted_fires": annoted_fires,
        "annoted_per_day": annoted_per_day,
        "unlabeled": unlabeled,
        "false_positives": false_positives,
        "false_positives_per_day": false_positives_per_day,
        "true_positives": true_positives,
        "true_positives_per_day": true_positives_per_day,
        "detections_per_hour": detections_per_hour,
        "detections_per_hour_site": detections_per_hour_site
    }

    return detections_per_day, stats

def plots(dataset, detections_per_day, stats):
    '''This function handles most of the plots'''
    annoted_per_day = stats['annoted_per_day']
    FP_per_day = stats['false_positives_per_day']
    TP_per_day = stats['true_positives_per_day']
    detections_per_hour = stats['detections_per_hour']
    detections_per_hour_site = stats['detections_per_hour_site']  

    # Total of detections  
    plt.figure(figsize=(12,6))
    detections_per_day.plot(kind='line', marker='o', label='Toutes les détections')
    annoted_per_day.plot(kind='line', marker='x', label='Alertes annotées')
    plt.title("Détections totales vs alertes annotées")
    plt.xlabel("Date")
    plt.ylabel("Nombre de détections")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # False positives 
    plt.figure(figsize=(12,6))
    FP_per_day.plot(kind='line', marker='s', color='red', label='Faux positifs')
    plt.title("Faux positifs par jour")
    plt.xlabel("Date")
    plt.ylabel("Nombre de faux positifs")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Number of fires, true positives 
    plt.figure(figsize=(12,6))
    TP_per_day.plot(kind='line', marker='o', color='green', label='Feux confirmés')
    plt.title("Nombre de feux confirmés par jour")
    plt.xlabel("Date")
    plt.ylabel("Nombre de feux")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # variation per hours for all sites 
    plt.figure(figsize=(12,6))
    detections_per_hour.plot(kind='bar', color='blue', alpha=0.7)
    plt.title("Variation des détections dans la journée (par heure)")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de détections")
    plt.tight_layout()
    plt.show()

    # hourly variation per site 
    plt.figure(figsize=(14,6))
    for site in detections_per_hour_site.index:
        plt.plot(
            detections_per_hour_site.columns,
            detections_per_hour_site.loc[site],
            marker="o",
            label=site
        )
    plt.title("Variations des détections dans la journée selon le site")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de détections")
    plt.legend(title="Site")
    plt.tight_layout()
    plt.show()

def summary_table(stats):
    '''Return a summary table: labeled,TP,FP,unlabeled'''
    df_summary = pd.DataFrame({
        "Total": [
            stats['annoted_fires'],
            stats['true_positives'],
            stats['false_positives'],
            stats['unlabeled']
        ]
    }, index=['Alertes annotées', 'Vrais positifs', 'Faux positifs', 'Non annotées'])
    return df_summary

def make_report_figures(texts,dataset, detections_per_day, stats, outdir="figs_tmp"):
    os.makedirs(outdir, exist_ok=True)
    paths = []

    fig_path = os.path.join(outdir, "detections_vs_annoted.png")
    plt.figure(figsize=(12,6))
    detections_per_day.plot(kind='line', marker='o', label='Toutes les détections')
    stats['annoted_per_day'].plot(kind='line', marker='x', label='Alertes annotées')
    plt.title(texts["titles"]["daily_detections"])
    plt.xlabel("Date"); plt.ylabel("Nombre de détections"); plt.xticks(rotation=45)
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    fig_path = os.path.join(outdir, "false_positives_per_day.png")
    plt.figure(figsize=(12,6))
    stats['false_positives_per_day'].plot(kind='line', marker='s', label='Faux positifs')
    plt.title(texts["titles"]["false_positives"])
    plt.xlabel("Date"); plt.ylabel("Nombre de faux positifs"); plt.xticks(rotation=45)
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    fig_path = os.path.join(outdir, "true_positives_per_day.png")
    plt.figure(figsize=(12,6))
    stats['true_positives_per_day'].plot(kind='line', marker='o', label='Feux confirmés', color='green')
    plt.title(texts["titles"]["confirmed_fires"])
    plt.xlabel("Date"); plt.ylabel("Nombre de feux"); plt.xticks(rotation=45)
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    # hourly variation 
    fig_path = os.path.join(outdir, "detections_per_hour.png")
    plt.figure(figsize=(12,6))
    stats['detections_per_hour'].plot(kind='bar')
    plt.title(texts["titles"]["detections_by_hour"])
    plt.xlabel("Heure"); plt.ylabel("Nombre de détections")
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    true_pos = dataset[dataset['is_wildfire'] == True].groupby('date')['id'].count()
    false_pos = dataset[dataset['is_wildfire'] == False].groupby('date')['id'].count()
    unlabeled = dataset[dataset['is_wildfire'].isna()].groupby('date')['id'].count()
    df_stack = pd.DataFrame({
        'True Positives': true_pos, 'False Positives': false_pos, 'Unlabeled': unlabeled
    }).fillna(0)
    fig_path = os.path.join(outdir, "stacked_by_day.png")
    ax = df_stack.plot(kind='bar', stacked=True, figsize=(12,6))
    step = max(1, len(df_stack)//15)  
    ax.set_xticks(range(0, len(df_stack), step))
    ax.set_xticklabels(df_stack.index[::step], rotation=45)
    plt.title(texts["titles"]["alerts_distribution"])
    plt.xlabel("Date"); plt.ylabel("Nombre d'alertes")
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    # hourly variation per site 
    fig_path = os.path.join(outdir, "detections_per_hour_site.png")
    plt.figure(figsize=(14,6))
    for site in stats['detections_per_hour_site'].index:
        plt.plot(
            stats['detections_per_hour_site'].columns,
            stats['detections_per_hour_site'].loc[site],
            marker="o",
            label=site
        )
    plt.title(texts["titles"]["site_hourly_comparison"])
    plt.xlabel("Heure de la journée"); plt.ylabel("Nombre de détections")
    plt.legend(title="Site"); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    paths.append(fig_path)

    return paths

def build_pdf_skeleton(texts,output_path, sdis_name,
                       nb_days=0, date_beginning="", date_end="", nb_cameras=0):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=48, bottomMargin=48
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Subtitle', parent=styles['Heading2'], spaceAfter=12, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='Comment', parent=styles['Normal'], fontSize=10, textColor=colors.darkgreen, spaceAfter=6))

    story = []

    story.append(Paragraph(texts["report_intro"]
                           .replace("${nom_sdis}", sdis_name)
                           .replace("${nb_days}", str(nb_days))
                           .replace("${date_beginning}", date_beginning)
                           .replace("${date_end}", date_end)
                           .replace("${nb_cameras}", str(nb_cameras)),
                           styles['Normal']))
    story.append(Spacer(1, 12))
    return doc, styles, story

def add_plot(story, doc, img_path, caption=None, styles=None):
    iw, ih = ImageReader(img_path).getSize()
    max_w = doc.width
    scale = max_w / float(iw)
    img = Image(img_path, width=max_w, height=ih*scale)

    elems = [img]
    if caption and styles:
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(caption, styles["Italic"]))
    elems.append(Spacer(1, 18))

    story.append(KeepTogether(elems))

def dataset_to_table(dataset, title="Synthèse"):
    data = [ ["" ] + list(dataset.columns) ] 
    for idx, row in dataset.iterrows():
        data.append([idx] + list(row.values))

    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    return tbl

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    text = f"Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M')} — Page {canvas.getPageNumber()}"
    canvas.drawRightString(doc.pagesize[0]-doc.rightMargin, doc.bottomMargin-10, text)
    canvas.restoreState()


def export_pdf_report(texts,output_path, dataset, detections_per_day, stats, df_summary, fig_paths, sdis_name ):
    nb_days, nb_cameras, nb_sites = days_and_cameras(dataset)
    dmin = str(min(dataset['date']))
    dmax = str(max(dataset['date']))

    # base
    doc, styles, story = build_pdf_skeleton(
    texts,
    output_path=output_path,
    sdis_name=sdis_name,
    nb_days=nb_days,
    date_beginning=dmin,
    date_end=dmax,
    nb_cameras=nb_cameras
)

    # title
    story.insert(0, Paragraph(f"Rapport Pyronear – {sdis_name}", styles["Title"]))
    story.insert(1, Spacer(1, 12))

    # table
    story.append(Paragraph("Tableau de synthèse des détections", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(dataset_to_table(df_summary))
    story.append(Spacer(1, 12))

    # all plots
    figure_info = [
        (fig_paths[0], texts["daily_detections"].replace("${nb_days}", str(nb_days))),
        (fig_paths[1], texts["false_positives"].replace("${date_beginning}", dmin).replace("${date_end}", dmax)),
        (fig_paths[2], texts["confirmed_fires"].replace("${nom_sdis}", sdis_name)),
        (fig_paths[4], texts["alerts_distribution"].replace("${nb_days}", str(nb_days))),
        (fig_paths[3], texts["detections_by_hour"].replace("${nb_cameras}", str(nb_cameras))),
        (fig_paths[5], texts["site_hourly_comparison"].replace("${nom_sdis}", sdis_name))
    ]

    for path, caption in figure_info:
        add_plot(story, doc, path, caption=caption, styles=styles)

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

def main():

    parser= argparse.ArgumentParser(description="Automatic performances report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_csv", type=str, default="pyronear.csv", help="Export file path")
    parser.add_argument("--text_path", type=str, default="text_pyronear.json", help="Repport text file")
    parser.add_argument("--output_pdf", type=str, default="pyronear_report.pdf", help="Output PDF")
    parser.add_argument("--sdis_name", type=str, default=None, help="SDIS name")

    args = parser.parse_args()

    texts = load_text_file(args.text_path)
    dataset = pd.read_csv(args.input_csv)
    dataset = date_to_datetime_grouped(dataset)
    detections_per_day, stats = first_statistics(dataset)
    df_summary = summary_table(stats)

    fig_paths = make_report_figures(texts,dataset, detections_per_day, stats)  
    export_pdf_report(texts,args.output_pdf, dataset, detections_per_day, stats, df_summary, fig_paths, sdis_name=args.sdis_name)

if __name__ == "__main__":
    main()
    
