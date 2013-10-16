using Accord.Controls.Imaging;
using Accord.Imaging;
using Accord.MachineLearning;
using Accord.Math.Geometry;
using AForge.Imaging;
using AForge.Imaging.Filters;
using AForge.Math.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            Threshold thresh = new Threshold(10);
            Median median = new Median(9);
            Erosion3x3 erode = new Erosion3x3();
            Dilatation3x3 dilate = new Dilatation3x3();
            GrahamConvexHull hullFinder = new GrahamConvexHull();
            ConnectedComponentsLabeling ccLabeler = new ConnectedComponentsLabeling();
            BorderFollowing contourFinder = new BorderFollowing();
            GrayscaleToRGB rgb = new GrayscaleToRGB();
            ConvexHullDefects defectFinder = new ConvexHullDefects(10);

            Bitmap img = (Bitmap)Bitmap.FromFile("hand3.jpg");

            Bitmap image = Grayscale.CommonAlgorithms.BT709.Apply(img);
            thresh.ApplyInPlace(image);
            //median.ApplyInPlace(image);
            erode.ApplyInPlace(image);
            dilate.ApplyInPlace(image);
            
            BlobCounter counter = new BlobCounter(image);
            counter.ObjectsOrder = ObjectsOrder.Area;

            Blob[] blobs = counter.GetObjectsInformation();

            if (blobs.Length > 0)
            {
                counter.ExtractBlobsImage(image, blobs[0], true);

                UnmanagedImage hand = blobs[0].Image;

                var contour = contourFinder.FindContour(hand);

                if (contour.Count() > 0)
                {
                    var initialHull = hullFinder.FindHull(contour);

                    var defects = defectFinder.FindDefects(contour, initialHull);

                    var filteredHull = initialHull.ClusterHullPoints().FilterLinearHullPoints();

                    var palmCenter = defects.Centroid(contour);

                    var wristPoints = filteredHull.SelectWristPoints(defects, contour);

                    Bitmap color = rgb.Apply(hand).ToManagedImage();

                    //BitmapData data = color.LockBits(new Rectangle(0, 0, color.Width, color.Height), ImageLockMode.ReadWrite, color.PixelFormat);
                    //Drawing.Polyline(data, contour, Color.Blue);
                    //Drawing.Polygon(data, filteredHull, Color.Red);
                    //color.UnlockBits(data);

                    Graphics gr = Graphics.FromImage(color);

                    gr.DrawPolygon(new Pen(Brushes.Red, 3), filteredHull.ToPtArray());
                    gr.DrawLines(new Pen(Brushes.Blue, 3), contour.ToPtArray());
                    gr.DrawEllipse(new Pen(Brushes.Red, 3), palmCenter.X - 10, palmCenter.Y - 10, 20, 20);

                    foreach (ConvexityDefect defect in defects)
                    {
                        gr.DrawEllipse(new Pen(Brushes.Green, 6), contour[defect.Point].X - 10, contour[defect.Point].Y - 10, 20, 20);
                    }

                    foreach (AForge.IntPoint pt in filteredHull)
                    {
                        gr.DrawEllipse(new Pen(Brushes.Yellow, 6), pt.X - 10, pt.Y - 10, 20, 20);
                    }

                    foreach (AForge.IntPoint pt in wristPoints)
                    {
                        gr.DrawEllipse(new Pen(Brushes.PowderBlue, 6), pt.X - 10, pt.Y - 10, 20, 20);
                    }

                    ImageBox.Show(color);
                }
            }
        }
    }

    public static class ListExtensions
    {
        public static Point[] ToPtArray(this List<AForge.IntPoint> list)
        {
            Point[] arr = new Point[list.Count()];

            for(int i = 0; i < list.Count(); ++i)
                arr[i] = new Point(list[i].X, list[i].Y);

            return arr;
        }

        public static List<AForge.IntPoint> FilterLinearHullPoints(this List<AForge.IntPoint> points)
        {
            AForge.IntPoint A, B, C;

            for (int i = 0; i < points.Count; ++i)
            {
                A = points[i];
                B = points[(i + 1) % points.Count];
                C = points[(i + 2) % points.Count];

                double top = Math.Abs((C.X - A.X) * (A.Y - B.Y) - (A.X - B.X) * (C.Y - A.Y));
                double bot = Math.Sqrt((C.X - A.X) * (C.X - A.X) + (C.Y - A.Y) * (C.Y - A.Y));
                double distance = top / bot;

                if (distance < 15)
                    points.RemoveAt((i + 1) % points.Count);
            }
            
            return points;
        }

        public static List<AForge.IntPoint> ClusterHullPoints(this List<AForge.IntPoint> points)
        {
            List<Cluster> Clusters = new List<Cluster>();

            while (points.Count > 0)
            {
                AForge.IntPoint pt = points[0];
                points.RemoveAt(0);

                Cluster cluster = new Cluster(pt);

                points.RemoveAll((p) =>
                {
                    if (cluster.CheckPoint(p))
                    {
                        cluster.Add(p);
                        return true;
                    }

                    return false;
                });

                Clusters.Add(cluster);
            }

            return Clusters.Centroids();
        }

        public static List<AForge.IntPoint> Centroids (this List<Cluster> clusters)
        {
            List<AForge.IntPoint> points = new List<AForge.IntPoint>();

            foreach (Cluster cluster in clusters)
            {
                points.Add(cluster.Centroid);
            }

            return points;
        }

        public static AForge.IntPoint Centroid (this List<AForge.IntPoint> points)
        {
            return points.Aggregate(new { xSum = 0, ySum = 0, n = 0 },
                                        (acc, p) => new
                                        {
                                            xSum = acc.xSum + p.X,
                                            ySum = acc.ySum + p.Y,
                                            n = acc.n + 1
                                        },
                                        acc => new AForge.IntPoint(acc.xSum / acc.n, acc.ySum / acc.n));
        }

        public static AForge.IntPoint Centroid(this List<ConvexityDefect> defects, List<AForge.IntPoint> contour)
        {
            List<AForge.IntPoint> defectPts = new List<AForge.IntPoint>();

            foreach (ConvexityDefect defect in defects)
            {
                defectPts.Add(contour[defect.Point]);
            }

            return defectPts.Centroid();
        }

        public static List<AForge.IntPoint> SelectWristPoints(this List<AForge.IntPoint> hull, List<ConvexityDefect> defects, List<AForge.IntPoint> contour)
        {
            List<AForge.IntPoint> defectPts = new List<AForge.IntPoint>();

            foreach (ConvexityDefect defect in defects)
            {
                defectPts.Add(contour[defect.Point]);
            }

            return hull.FindAll((p1) =>
            {
                foreach (AForge.IntPoint p2 in defectPts)
                {
                    if (p1.Y < p2.Y)
                        return false;
                }

                return true;
            });
        }
    }

    public class Cluster
    {
        public Cluster(AForge.IntPoint initialPt)
        {
            Points.Add(initialPt);
        }

        List<AForge.IntPoint> Points = new List<AForge.IntPoint>();

        public AForge.IntPoint Centroid
        {
            get
            {
                return Points.Centroid();
            }
        }

        public void Add(AForge.IntPoint pt)
        {
            Points.Add(pt);
        }

        public bool CheckPoint(AForge.IntPoint pt)
        {
            float radius = 20.0f;

            return (Points.FindAll((p) =>
                    {
                        if (p.DistanceTo(pt) <= 2 * radius)
                        {
                            return true;
                        }

                        return false;
                    }).Count > 0);
        }
    }
}
